-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NDArray
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- NDArray module, provide an imperative-style programming interface.
--
{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# OPTIONS_GHC -Wno-unused-do-bind #-}

{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}

module MXNet.Core.NDArray where

import           Control.Monad
import           Data.Int
import           Data.Monoid
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           Foreign.Marshal.Alloc (alloca)
import           Foreign.Marshal.Array (peekArray)
import           Foreign.Ptr
import           Foreign.Storable (Storable)
import           GHC.Exts (IsList(..))
import           Text.PrettyPrint.Annotated.HughesPJClass (Pretty(..), prettyShow)
import           System.IO.Unsafe (unsafePerformIO)

import           MXNet.Core.Base
import qualified MXNet.Core.Base.Internal.TH.NDArray as I
import           MXNet.Core.HMap

-- | NDArray type alias.
newtype NDArray a = NDArray { getHandle :: NDArrayHandle }

-- | DType class, used to quantify types that can be passed to mxnet.
class (Storable a, Show a, Eq a, Ord a, Num a, Real a) => DType a where
    typeid :: NDArray a -> Int
    typename :: NDArray a -> String

pattern FLOAT32 = 0
pattern FLOAT64 = 1
pattern FLOAT16 = 2
pattern UINT8   = 3
pattern INT32   = 4

instance DType Float where
    typeid _ = FLOAT32
    typename _ = "float32"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Double where
    typeid _ = FLOAT32
    typename _ = "float64"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Int8 where
    typeid _ = UINT8
    typename _ = "uint8"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Int32 where
    typeid _ = INT32
    typename _ = "int32"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

-- | Wrapper for pretty print multiple dimensions matrices.
data PrettyWrapper = forall a. Pretty a => MkPretty { runPretty :: a }

instance Pretty PrettyWrapper where
    pPrint (MkPretty inner) = pPrint inner

instance (DType a, Pretty a) => Show (NDArray a) where
    -- TODO display more related information.
    show array = unsafePerformIO $ do
        (_, dims) <- shape array
        values <- items array
        let info = show dims
            body = prettyShow . splitItems values dims $ 0
        return ("NDArray " <> info <> "\n" <> body)
      where
        splitItems :: Vector a -> [Int] -> Int -> PrettyWrapper
        splitItems _ [] _ = error "Impossible: never match an empty list."
        splitItems values [x] s = MkPretty . toList $ V.unsafeSlice s x values
        splitItems values (d:ds) s = MkPretty $ (\x -> splitItems values ds (s + (product ds) * x)) <$> ([0 .. (d - 1)] :: [Int])

instance DType a => Num (NDArray a) where
    (+) arr1 arr2 = NDArray $ unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_add handle1 handle2
    (-) arr1 arr2 = NDArray $ unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_sub handle1 handle2
    (*) arr1 arr2 = NDArray $ unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_mul handle1 handle2
    abs arr1 = NDArray $ unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.abs handle1
    negate arr1 = NDArray $ unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.negative handle1
    signum = error "Unsupported operation: signum(NDArray)"
    fromInteger = error "Unsupported operation: fromInteger(NDArray)"

-- | Context definition.
--
--      * DeviceType
--
--          1. cpu
--          2. gpu
--          3. cpu_pinned
data Context = Context { deviceType :: Int
                       , deviceId   :: Int
                       } deriving (Eq, Show)

-- | Default context, use the CPU 0 as device.
defaultContext :: Context
defaultContext = Context { deviceType = 1   -- cpu
                         , deviceId = 1     -- default value.
                         }

-- | Context for CPU 0.
contextCPU :: Context
contextCPU = Context 1 0

-- | Context for GPU 0.
contextGPU :: Context
contextGPU = Context 2 0

-- | Wait all async operation to finish in MXNet.
waitAll :: IO ()
waitAll = void mxNDArrayWaitAll

-- | Make a new empty ndarray with specified shape, context and data type.
makeEmptyNDArray :: forall a. DType a
                 => [Int]           -- ^ Shape.
                 -> Context         -- ^ Context/
                 -> Bool            -- ^ If delayed allocate.
                 -> IO (NDArray a)
makeEmptyNDArray sh ctx delayed = do
    let sh' = fromIntegral <$> sh
        nlen = fromIntegral . length $ sh
        dtype = typeid (undefined :: NDArray a)
    (_, handle) <- mxNDArrayCreateEx sh' nlen (deviceType ctx) (deviceId ctx) (if delayed then 1 else 0) dtype
    return $ NDArray handle

-- | Make a new NDArray with given shape.
makeNDArray :: DType a
            => [Int]            -- ^ size of every dimensions.
            -> Context
            -> Vector a
            -> IO (NDArray a)
makeNDArray sh ctx ds = do
    let sh' = fromIntegral <$> sh
        nlen = fromIntegral . length $ sh
    (_, handle) <- mxNDArrayCreate sh' nlen (deviceType ctx) (deviceId ctx) 0
    V.unsafeWith ds $ \p -> do
        let len = fromIntegral (V.length ds)
        mxNDArraySyncCopyFromCPU handle (castPtr p) len
    return $ NDArray handle

-- | Get the shape of given NDArray.
shape :: DType a
      => NDArray a
      -> IO (Int, [Int])  -- ^ Dimensions and size of every dimensions.
shape arr = do
    (_, nlen, sh) <- mxNDArrayGetShape (getHandle arr)
    return (fromIntegral nlen, fromIntegral <$> sh)

-- | Get size of the given ndarray.
size :: DType a
     => NDArray a
     -> IO Int      -- ^ Dimensions and size of every dimensions.
size arr = (product . snd) <$> shape arr

-- | Get context of the given ndarray.
context :: DType a => NDArray a -> IO Context
context arr = do
    (_, device'type, device'id) <- mxNDArrayGetContext (getHandle arr)
    return $ Context device'type device'id

-- | Make a copy of the give ndarray.
copy :: DType a => NDArray a -> IO (NDArray a)
copy arr = NDArray <$> I._copy (getHandle arr)

-- | Get data stored in NDArray.
items :: DType a => NDArray a -> IO (Vector a)
items arr = do
    nlen <- (product . snd) <$> shape arr
    alloca $ \p -> do
        mxNDArraySyncCopyToCPU (getHandle arr) p (fromIntegral nlen)
        fromList <$> peekArray nlen (castPtr p :: Ptr a)

-- | Return a sliced ndarray that __shares memory__ with current one.
slice :: DType a
      => NDArray a
      -> Int        -- ^ The beginning index of slice.
      -> Int        -- ^ The end index of slices.
      -> NDArray a
slice arr start end = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArraySlice handle (fromIntegral start) (fromIntegral end)
    return handle'

-- | Return a sub ndarray that __shares memory__ with current one.
at :: DType a
   => NDArray a
   -> Int       -- ^ The index.
   -> NDArray a
at arr idx = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArrayAt handle (fromIntegral idx)
    return handle'

-- | Return a reshaped ndarray that __shares memory__ with current one.
reshape :: DType a
        => NDArray a
        -> [Int]        -- ^ Size of every dimension of new shape.
        -> NDArray a
reshape arr sh = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArrayReshape handle (length sh) sh
    return handle'

-- | Block until all pending writes operations on current ndarray are finished.
waitToRead :: DType a => NDArray a -> IO ()
waitToRead arr = void $ mxNDArrayWaitToRead (getHandle arr)

-- | One hot encoding indices into matrix out.
onehotEncode :: DType a
             => NDArray a       -- ^ An ndarray containing indices of the categorical features.
             -> NDArray a       -- ^ The result holder of the encoding.
             -> IO (NDArray a)  -- ^ The encoding ndarray.
onehotEncode indices out = do
    let handle1 = getHandle indices
        handle2 = getHandle out
    NDArray <$> I._onehot_encode' handle1 handle2 [handle2]

-- | Create a new NDArray filled with 0, with specified shape and context.
zeros :: DType a
      => [Int]      -- ^ Shape.
      -> IO (NDArray a)
zeros sh = full sh 0

-- | Create a new NDArray filled with 1, with specified shape and context.
ones :: DType a
     => [Int]      -- ^ Shape.
     -> IO (NDArray a)
ones sh = full sh 1

-- | Create a new NDArray filled with given value, with specified shape and context.
full :: DType a
      => [Int]      -- ^ Shape.
      -> a          -- ^ Given value to fill the ndarray.
      -> IO (NDArray a)
full sh value = makeNDArray sh defaultContext $ V.replicate (product sh) value

-- | Create a new NDArray that copies content from source_array.
array :: DType a
      => [Int]      -- ^ Shape.
      -> Vector a
      -> IO (NDArray a)
array sh = makeNDArray sh defaultContext

-- | Add a scalar to a narray.
(.+) :: DType a
      => NDArray a -> a -> NDArray a
(.+) arr value = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._plus_scalar handle (realToFrac value)

infixl 6 .+

-- | Mutably add a scalar to a narray, the argument ndarray will be modified as the result.
(.+=) :: DType a
      => NDArray a -> a -> IO ()
(.+=) arr value = do
    let handle = getHandle arr
    I._plus_scalar' handle (realToFrac value) [handle]

{-# INLINE (.+) #-}

-- | Subtract a scalar from a narray.
(.-) :: DType a
      => NDArray a -> a -> NDArray a
(.-) arr value = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._minus_scalar handle (realToFrac value)

infixl 6 .-

{-# INLINE (.-) #-}

-- | Subtract a scalar from a ndarray, the argument ndarray will be modified as the result.
(.-=) :: DType a
      => NDArray a -> a -> IO ()
(.-=) arr value = do
    let handle = getHandle arr
    I._minus_scalar' handle (realToFrac value) [handle]

-- | Multiply a scalar to a narray.
(.*) :: DType a
      => NDArray a -> a -> NDArray a
(.*) arr value = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._mul_scalar handle (realToFrac value)

infixl 7 .*

{-# INLINE (.*) #-}

-- | Multiply a scalar to a ndarray, the argument ndarray will be modified as the result.
(.*=) :: DType a
      => NDArray a -> a -> IO ()
(.*=) arr value = do
    let handle = getHandle arr
    I._mul_scalar' handle (realToFrac value) [handle]

-- | Divide a scalar from a narray.
(./) :: DType a
      => NDArray a -> a -> NDArray a
(./) arr value = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._div_scalar handle (realToFrac value)

infixl 7 ./

{-# INLINE (./) #-}

-- | Divide a scalar from a ndarray, the argument ndarray will be modified as the result.
(./=) :: DType a
      => NDArray a -> a -> IO ()
(./=) arr value = do
    let handle = getHandle arr
    I._div_scalar' handle (realToFrac value) [handle]

-- | Power of a ndarray, use a scalar as exponent.
(.^) :: DType a
      => NDArray a -> a -> NDArray a
(.^) arr value = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._power_scalar handle (realToFrac value)

infixl 8 .^

{-# INLINE (.^) #-}

-- | Power of a ndarray, use a scalar as exponent, the argument ndarray will be modified as the result.
(.^=) :: DType a
      => NDArray a -> a -> IO ()
(.^=) arr value = do
    let handle = getHandle arr
    I._power_scalar' handle (realToFrac value) [handle]

-- | Power of a narray, use a ndarray as exponent.
(..^) :: DType a
      => a -> NDArray a -> NDArray a
(..^) value arr = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._rpower_scalar handle (realToFrac value)

infixl 8 ..^

{-# INLINE (..^) #-}

-- | Power of a ndarray, use a ndarray as exponent, the argument ndarray will be modified as the result.
(..^=) :: DType a
       => a -> NDArray a -> IO ()
(..^=) value arr = do
    let handle = getHandle arr
    I._power_scalar' handle (realToFrac value) [handle]

-- | Maximum elements in the given two ndarrays.
maximum :: DType a
        => NDArray a -> NDArray a -> NDArray a
maximum arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_maximum handle1 handle2

{-# INLINE maximum #-}

-- | Maximum elements in the given ndarray and given scalar.
maximum' :: DType a
        => NDArray a -> a -> NDArray a
maximum' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._maximum_scalar handle (realToFrac scalar)

{-# INLINE maximum' #-}

-- | Minimum elements in the given two ndarrays.
minimum :: DType a
        => NDArray a -> NDArray a -> NDArray a
minimum arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_minimum handle1 handle2

{-# INLINE minimum #-}

-- | Minimum elements in the given ndarray and given scalar.
minimum' :: DType a
        => NDArray a -> a -> NDArray a
minimum' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._minimum_scalar handle (realToFrac scalar)

{-# INLINE minimum' #-}

-- | If elements in the given two ndarrays are equal.
equal :: DType a
      => NDArray a -> NDArray a -> NDArray a
equal arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_equal handle1 handle2

{-# INLINE equal #-}

-- | If elements in the given ndarray are equal to the given scalar.
equal' :: DType a
       => NDArray a -> a -> NDArray a
equal' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._equal_scalar handle (realToFrac scalar)

{-# INLINE equal' #-}

-- | If elements in the given two ndarrays are not equal.
notEqual :: DType a
          => NDArray a -> NDArray a -> NDArray a
notEqual arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_not_equal handle1 handle2

{-# INLINE notEqual #-}

-- | If elements in the given ndarray are equal to the given scalar.
notEqual' :: DType a
        => NDArray a -> a -> NDArray a
notEqual' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._not_equal_scalar handle (realToFrac scalar)

{-# INLINE notEqual' #-}

-- | If elements in the first given ndarrays are greater than the second one.
greater :: DType a
        => NDArray a -> NDArray a -> NDArray a
greater arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_greater handle1 handle2

{-# INLINE greater #-}

-- | If elements in the first given ndarrays are greater than the give scalar.
greater' :: DType a
        => NDArray a -> a -> NDArray a
greater' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._greater_scalar handle (realToFrac scalar)

{-# INLINE greater' #-}

-- | If elements in the first given ndarrays are greater than or equal to the second one.
greaterEqual :: DType a
        => NDArray a -> NDArray a -> NDArray a
greaterEqual arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_greater_equal handle1 handle2

{-# INLINE greaterEqual #-}

-- | If elements in the first given ndarrays are greater than or equal to the give scalar.
greaterEqual' :: DType a
              => NDArray a -> a -> NDArray a
greaterEqual' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._greater_equal_scalar handle (realToFrac scalar)

{-# INLINE greaterEqual' #-}

-- | If elements in the first given ndarrays are lesser than the second one.
lesser :: DType a
       => NDArray a -> NDArray a -> NDArray a
lesser arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_lesser handle1 handle2

{-# INLINE lesser #-}

-- | If elements in the first given ndarrays are lesser than the give scalar.
lesser' :: DType a
        => NDArray a -> a -> NDArray a
lesser' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._lesser_scalar handle (realToFrac scalar)

{-# INLINE lesser' #-}

-- | If elements in the first given ndarrays are lesser than or equal to the second one.
lesserEqual :: DType a
            => NDArray a -> NDArray a -> NDArray a
lesserEqual arr1 arr2 = NDArray . unsafePerformIO $ do
    let handle1 = getHandle arr1
        handle2 = getHandle arr2
    I.broadcast_lesser_equal handle1 handle2

{-# INLINE lesserEqual #-}

-- | If elements in the first given ndarrays are lesser than or equal to the give scalar.
lesserEqual' :: DType a
             => NDArray a -> a -> NDArray a
lesserEqual' arr scalar = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    I._lesser_equal_scalar handle (realToFrac scalar)

{-# INLINE lesserEqual' #-}

{-------------------------------------------------------------------------------
-- Neural Network
-------------------------------------------------------------------------------}

-- | Apply a linear transformation: /Y = X W^T + b/.
fullyconnected :: DType a
               => NDArray a -- ^ Input data.
               -> NDArray a -- ^ Weight matrix.
               -> NDArray a -- ^ Bias parameter.
               -> Int       -- ^ Number of hidden nodes of the output.
               -> NDArray a
fullyconnected input weight bias n = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.fullyconnected handle1 handle2 handle3 n nil

-- Convolution Compute N-D convolution on (N+2)-D input.
convolution :: DType a
            => NDArray a    -- ^ Input data.
            -> NDArray a    -- ^ Weight matrix.
            -> NDArray a    -- ^ Bias parameter.
            -> String       -- ^ Convolution kernel size: (h, w) or (d, h, w).
            -> Int          -- ^ Convolution filter(channel) number.
            -> NDArray a
convolution input weight bias kernel n = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.convolution handle1 handle2 handle3 kernel n nil

-- | Elementwise activation function.
activation :: DType a
           => NDArray a -- ^ Input data to activation function.
           -> String    -- ^ Activation function to be applied, one of {'relu', 'sigmoid', 'softrelu', 'tanh'}.
           -> NDArray a
activation input act = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
    I.activation handle1 act

-- | Batch normalization.
batchnorm :: DType a
          => NDArray a  -- ^ Input data to batch normalization.
          -> NDArray a  -- ^ Gamma array.
          -> NDArray a  -- ^ Beta array.
          -> NDArray a  
batchnorm input weight bias = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.batchnorm handle1 handle2 handle3 nil

-- | Perform pooling on the input.
pooling :: DType a
        => NDArray a    -- ^ Input data to the pooling operator.
        -> String       -- ^ Pooling kernel size: (y, x) or (d, y, x).
        -> String       -- ^ Pooling type to be applied, one of {'avg', 'max', 'sum'}.
        -> NDArray a
pooling input kernel pooltype = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
    I.pooling handle1 kernel pooltype nil

-- | Softmax with logit loss.
softmaxoutput :: DType a
              => NDArray a  -- ^ Input data.
              -> NDArray a  -- ^ Ground truth label.
              -> NDArray a
softmaxoutput input label = NDArray . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle label
    I.softmaxoutput handle1 handle2 nil

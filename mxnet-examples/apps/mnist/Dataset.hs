{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
module Dataset where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Symbol  as S
import qualified MXNet.Core.Base.Internal.TH.NDArray as MXI
import Data.Function ((&))
import Streaming
import Streaming.Prelude (Of(..))
import qualified Streaming.Prelude as S
import Control.Monad.Trans.Resource (ResourceT, MonadResource(..))
import qualified Data.ByteString as BS
import qualified Data.Vector as NV
import qualified Data.Vector.Storable as SV
import Control.Monad (liftM2)
import Control.Monad.IO.Class
import Data.List (unfoldr)
import GHC.TypeLits (natVal, KnownNat, Nat)
import Data.Proxy (Proxy(..))

import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

device :: Context
device = contextCPU

type StreamProc a b m = Stream (Of a) m () -> Stream (Of b) m ()

mappedOf :: Monad m => (a -> m b) -> StreamProc a b m
-- mappedOf f = S.sequence . maps (first f)
mappedOf = S.mapM

cImageToNDArray :: MonadIO m => StreamProc (Batched Image) ArrayF m
cImageToNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  makeNDArray [sz, 28, 28] device $ SV.concat $ NV.toList $ _batch dat

cLabelToOnehotNDArray :: MonadIO m => StreamProc (Batched Label) ArrayF m
cLabelToOnehotNDArray = mappedOf $ \dat -> liftIO $ do
  let sz = size dat
  a <- array [sz] (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF
  b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
  reshape (A.NDArray b) [sz, 10]

cBatchN :: MonadIO m => Int -> StreamProc a (Batched a) m
cBatchN n = mapped toBatch . chunksOf n
  where
    toBatch seg = first (Batched . NV.fromList) <$> S.toList seg

trainingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) m ()
trainingData = S.zip
    (sourceImages "mxnet-examples/data/train-images-idx3-ubyte" & cBatchN 32 & cImageToNDArray      )
    (sourceLabels "mxnet-examples/data/train-labels-idx1-ubyte" & cBatchN 32 & cLabelToOnehotNDArray)

newtype Batched a = Batched { _batch :: NV.Vector a }

size :: Batched a -> Int
size (Batched b) = NV.length b

-- cImageToNDArray :: MonadIO m => Batched (SV.Vector Float) -> m ArrayF
-- cImageToNDArray dat = liftIO $ do
--   let sz = size dat
--   makeNDArray [sz, 28, 28] device $ SV.concat $ NV.toList $ _batch dat

-- cLabelToOnehotNDArray :: MonadIO m => Batched Int -> m ArrayF
-- cLabelToOnehotNDArray dat = liftIO $ do
--   let sz = size dat
--   a <- array [sz] (NV.convert $ NV.map fromIntegral $ _batch dat) :: IO ArrayF
--   b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
--   reshape (A.NDArray b) [sz, 10]

-- cBatchN :: Int -> NV.Vector a -> NV.Vector (Batched a)
-- cBatchN n v = NV.fromList $ map Batched (chunksOf n v)
--   where
--     chunksOf :: Int -> NV.Vector a -> [NV.Vector a]
--     chunksOf i = unfoldr go
--       where go v | NV.null v = Nothing
--                  | otherwise = Just (NV.splitAt i v)

-- trainingData :: MonadResource m => m (NV.Vector (ArrayF, ArrayF))
-- trainingData = liftM2 NV.zip a b
--   where
--     a = (parseImages "mxnet-examples/data/train-images-idx3-ubyte" >>= NV.mapM cImageToNDArray       . cBatchN 32)
--     b = (parseLabels "mxnet-examples/data/train-labels-idx1-ubyte" >>= NV.mapM cLabelToOnehotNDArray . cBatchN 32)

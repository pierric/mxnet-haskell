{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
module Dataset where

import MXNet.Core.Base
import qualified MXNet.Core.Base.NDArray as A
-- import qualified MXNet.Core.Base.Symbol  as S
import qualified MXNet.Core.Base.Internal.TH.NDArray as MXI
import Data.Function ((&))
import Streaming
import Streaming.Prelude (Of(..))
import qualified Streaming.Prelude as S
import Control.Monad.Trans.Resource (ResourceT)

import Parse

type SymbolF = Symbol Float
type ArrayF  = NDArray Float

device :: Context
device = contextCPU

type StreamProc a b m = Stream (Of a) m () -> Stream (Of b) m ()

mappedOf :: Monad m => (a -> m b) -> StreamProc a b m
mappedOf f = S.sequence . maps (first f)

cImageToNDArray :: MonadIO m => StreamProc Image ArrayF m
cImageToNDArray = mappedOf (liftIO . makeNDArray [28,28] device)

cLabelToOnehotNDArray :: MonadIO m => StreamProc Label ArrayF m
cLabelToOnehotNDArray = mappedOf (\i -> liftIO $ do
     a <- array [1] [fromIntegral i] :: IO ArrayF
     b <- MXI.one_hot (A.getHandle a) 10 (add @"on_value" 1.0 $ add @"off_value" 0.0 nil)
     reshape (A.NDArray b) [10])

cBatchN :: MonadIO m => Int -> StreamProc ArrayF ArrayF m
cBatchN n = mappedOf stack . mapped S.toList . chunksOf n
  where
    stack window = do 
        let cnt = length window
        (_, sh) <- liftIO $ ndshape (head window)
        elem_raw  <- liftIO $ mapM (\a -> A.getHandle <$> reshape a (1:sh)) window
        outp_raw   <- liftIO $ MXI.concat elem_raw cnt (add @"dim" 0 nil)
        return (A.NDArray outp_raw)

trainingData :: MonadResource m => Stream (Of (ArrayF, ArrayF)) m ()
trainingData = S.zip
    (sourceImages "mxnet-examples/data/train-images-idx3-ubyte" & cImageToNDArray       & cBatchN 32)
    (sourceLabels "mxnet-examples/data/train-labels-idx1-ubyte" & cLabelToOnehotNDArray & cBatchN 32)

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
module Main where

import MXNet.Core.Base
import MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Executor as E
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Internal.TH.Symbol as S
import qualified Data.HashMap.Strict as M
import Data.List (intersperse)
import qualified Control.Monad.State as ST
import Data.Maybe (isNothing)
import Control.Monad (when, forM_)
import Control.Monad.IO.Class
import qualified Streaming.Prelude as SR
import qualified Streaming as SR
import Control.Monad.Trans.Resource
import Control.Monad.Morph (lift)
import qualified Data.Vector as NV

import Dataset

neural = do
    x  <- variable "x"  :: IO SymbolF 
    y  <- variable "y"  :: IO SymbolF
    w1 <- variable "w1" :: IO SymbolF
    b1 <- variable "b1" :: IO SymbolF
    a1 <- fullyConnected x w1 b1 128 >>= flip activation "relu"
    w2 <- variable "w2" :: IO SymbolF
    b2 <- variable "b2" :: IO SymbolF
    fullyConnected a1 w2 b2 10 >>= flip softmaxOutput y


data Param = Param { _param_in :: ArrayF, _param_grad :: ArrayF }
    deriving Show
type TrainM m = ST.StateT (Maybe (M.HashMap String Param)) m

train :: Monad m => TrainM m r -> m r
train = flip ST.evalStateT Nothing

initParam :: SymbolF -> M.HashMap String ArrayF -> IO (M.HashMap String Param)
initParam sym dat = do
    let (names, vals) = unzip $ M.toList dat
    shapes <- mapM ndshape vals
    let arg_ind = scanl (+) 0 $ map fst shapes
        arg_shp = concat $ map snd shapes
    (inp_shp, _, _) <- mxSymbolInferShape (S.getHandle sym) names arg_ind arg_shp
    inps <- listInputs sym
    let inp_with_shp = M.fromList $ zip inps inp_shp
    M.traverseWithKey init_with_random_normal inp_with_shp
  where
    init_with_random_normal inp shp = do
        grad <- makeEmptyNDArray shp device True
        case M.lookup inp dat of
            Just in_arg -> return $ Param in_arg grad
            Nothing -> do
                in_handle <- A.random_normal (add @"loc" 0 $ add @"scale" 1 $ add @"shape" (formatShape shp) nil)
                return $ Param (A.NDArray in_handle) grad
    formatShape shp = concat $ ["("] ++ intersperse "," (map show shp) ++ [")"]

trainStep :: MonadIO m => SymbolF -> M.HashMap String ArrayF -> TrainM m ()
trainStep net datAndLbl  = do
     uninited <- ST.gets isNothing
     when uninited (liftIO (initParam net datAndLbl) >>= ST.put . Just)
     Just params <- ST.get
     let params' = M.foldrWithKey (\k v -> M.adjust (\p -> p {_param_in = v}) k) params datAndLbl
     liftIO $ do
        exec <- bindParam net params'
        checked $ mxExecutorForward (E.getHandle exec) 1
        backward exec

bindParam :: SymbolF -> M.HashMap String Param -> IO (Executor Float)
bindParam net args = do
    names <- listInputs net
    exec_handle <- checked $ mxExecutorBind (S.getHandle net) (deviceType device) (deviceId device)
        (fromIntegral (M.size args))
        (map (A.getHandle . _param_in)   $ map (args M.!) names)
        (map (A.getHandle . _param_grad) $ map (args M.!) names)
        (replicate (M.size args) 1)
        0 []
    
    makeExecutor exec_handle

main = do
  _  <- mxListAllOpNames
  net <- neural
  runResourceT $ train $ go 3 net trainingData  

  where
    go = go' 0
    go' i n net dat
      | i < n = do liftIO $ putStrLn $ "iteration " ++ show i
                   SR.iterT SR.snd' $ SR.chain (\(dat, lbl) -> trainStep net $ M.fromList [("x", dat), ("y", lbl)]) dat
                   go' (i+1) n net dat
      | otherwise = return ()


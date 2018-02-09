{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE LambdaCase #-}
module Main where

import MXNet.Core.Base
import MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Executor as E
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Types.Internal as MXI
import qualified Data.HashMap.Strict as M
import Data.List (intersperse)
import qualified Control.Monad.State as ST
import Data.Maybe (isNothing, fromJust)
import Control.Monad (when, void, forM)
import Control.Monad.IO.Class
import qualified Streaming.Prelude as SR
import qualified Streaming as SR
import Control.Monad.Trans.Resource
import Control.Exception.Base
import Data.Typeable

import Dataset

neural :: IO SymbolF
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

inferShape :: SymbolF -> M.HashMap String ArrayF -> IO (M.HashMap String [Int])
inferShape sym known = do
    let (names, vals) = unzip $ M.toList known
    shapes <- mapM ndshape vals
    let arg_ind = scanl (+) 0 $ map fst shapes
        arg_shp = concat $ map snd shapes
    (inp_shp, _, _) <- mxSymbolInferShape (S.getHandle sym) names arg_ind arg_shp
    inps <- listInputs sym
    return $ M.fromList $ zip inps inp_shp

initParam :: SymbolF -> M.HashMap String ArrayF -> IO (M.HashMap String Param)
initParam sym dat = do
    inp_with_shp <- inferShape sym dat
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

buildParamFromShape :: [Int] -> IO Param
buildParamFromShape shp = do
    arr <- makeEmptyNDArray shp device True
    gra <- makeEmptyNDArray shp device True
    return $ Param arr gra

buildParam :: ArrayF -> IO Param
buildParam arr = do
    shp <- ndshape arr
    gra <- makeEmptyNDArray (snd shp) device True
    return $ Param arr gra

checkShapeUpd :: M.HashMap String ArrayF -> M.HashMap String ArrayF -> IO Bool
checkShapeUpd arr1 arr2 = do cs <- sequence $ M.intersectionWithKey chk arr1 arr2
                             return $ any id $ M.elems cs
  where
    chk _ a1 a2 = do
        s1 <- ndshape a1
        s2 <- ndshape a2
        return (s1 /= s2)

trainStep :: MonadIO m => SymbolF -> M.HashMap String ArrayF -> TrainM m ()
trainStep net datAndLbl = do
    uninited <- ST.gets isNothing
    when uninited (liftIO (initParam net datAndLbl) >>= ST.put . Just)
    Just params <- ST.get
    liftIO $ do 
        let params' = M.foldrWithKey (\k v -> M.adjust (\p -> p {_param_in = v}) k) params datAndLbl
        exec <- bindParam net params' True
        checked $ mxExecutorForward (E.getHandle exec) 1
        backward exec
        void $ flip M.traverseWithKey params' $ \ k v -> do
            when (not $ M.member k datAndLbl) $ 
                A.sgd_update (A.getHandle $ _param_in v) (A.getHandle $ _param_grad v) 0.001 nil

trainForwardOnly :: (MonadIO m, MonadThrow m) => SymbolF -> M.HashMap String ArrayF -> TrainM m [ArrayF]
trainForwardOnly net dat =
     ST.get >>= \case
        Nothing -> throwM NetworkUninitialized
        Just params -> liftIO $ do
            let params' = M.foldrWithKey (\k v -> M.adjust (\p -> p {_param_in = v}) k) params dat
            exec <- bindParam net params' False
            checked $ mxExecutorForward (E.getHandle exec) 0
            getOutputs exec

bindParam :: SymbolF -> M.HashMap String Param -> Bool -> IO (Executor Float)
bindParam net args train = do
    names <- listInputs net
    exec_handle <- checked $ mxExecutorBind (S.getHandle net) (deviceType device) (deviceId device)
        (fromIntegral (M.size args))
        -- the parameters to bind should be arranged in the same order as the names
        (map (A.getHandle . _param_in)   $ map (args M.!) names)
        (if train 
           then map (A.getHandle . _param_grad) $ map (args M.!) names
           else replicate (M.size args) MXI.nullNDArrayHandle)
        (replicate (M.size args) 1)
        0 []
    
    makeExecutor exec_handle

data Exc = NetworkUninitialized | UnknownLabelName String
    deriving (Show, Typeable)
instance Exception Exc

main :: IO ()
main = do
  _  <- mxListAllOpNames
  net <- neural
  runResourceT $ train $ do 
    liftIO $ putStrLn $ "[Train] "
    ST.forM_ (range 1) $ step net trainingData  
    liftIO $ putStrLn $ "[Test] "
    result <- SR.toList_ $ flip SR.mapM testingData $ \(x, y) -> do 
        [y'] <- trainForwardOnly net (M.singleton "x" x)
        return (y, y')
    liftIO $ print $ take 10 result

  where
    range :: Int -> [Int]
    range = enumFromTo 1
    step net dat ind = do
        liftIO $ putStrLn $ "iteration " ++ show ind
        SR.mapM_ (\(x, y) -> trainStep net $ M.fromList [("x", x), ("y", y)]) dat


{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE LambdaCase #-}
module Main where

import MXNet.Core.Base
import MXNet.Core.Base.Internal
import MXNet.Core.Types.Internal
import qualified MXNet.Core.Base.NDArray as A
import qualified MXNet.Core.Base.Symbol as S
import qualified MXNet.Core.Base.Executor as E
import qualified MXNet.Core.Base.Internal.TH.NDArray as A
import qualified MXNet.Core.Types.Internal as MXI
import qualified Data.HashMap.Strict as M
import Data.List (intersperse)
import qualified Control.Monad.State as ST
import Data.Maybe (isNothing, isJust, fromJust)
import Control.Monad (when)
import Control.Monad.IO.Class
import qualified Streaming.Prelude as SR
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
    v1 <- fullyConnected x w1 b1 128
    a1 <- activation v1 "relu"
    w2 <- variable "w2" :: IO SymbolF
    b2 <- variable "b2" :: IO SymbolF
    v2 <- fullyConnected a1 w2 b2 10
    a2 <- softmaxOutput v2 y 
    return a2

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

-- initialize all parameters, whose _in is sampled by a normal distr (0,1), and _grad is zeros.
initParam :: SymbolF -> M.HashMap String ArrayF -> IO (M.HashMap String Param)
initParam sym dat = do
    inp_with_shp <- inferShape sym dat
    M.traverseWithKey init_with_random_normal inp_with_shp
  where
    init_with_random_normal inp shp = do
        case M.lookup inp dat of
            Just in_arg -> return $ Param in_arg (A.NDArray nullNDArrayHandle)
            Nothing -> do
                in_handle <- A.random_normal (add @"loc" 0 $ add @"scale" 1 $ add @"shape" (formatShape shp) nil)
                gra <- zeros shp
                return $ Param (A.NDArray in_handle) gra
    formatShape shp = concat $ ["("] ++ intersperse "," (map show shp) ++ [")"]

-- bind the symbolic network with actual parameters
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

-- single step train. Must provide all the placeholders.
trainStep :: MonadIO m => SymbolF -> M.HashMap String ArrayF -> TrainM m ()
trainStep net datAndLbl = do
    uninited <- ST.gets isNothing
    when uninited (liftIO (initParam net datAndLbl) >>= ST.put . Just)
    Just params <- ST.get
    params <- flip M.traverseWithKey params (\k p -> do
        case M.lookup k datAndLbl of
            Just a  -> return $ p {_param_in = a}
            Nothing -> return p)
    liftIO $ do 
        exec <- bindParam net params True
        checked $ mxExecutorForward (E.getHandle exec) 1
        backward exec
    params <- flip M.traverseWithKey params $ \ k v -> do
                if (not $ M.member k datAndLbl) 
                    then do
                        new_in <- A.NDArray <$> liftIO (A.sgd_update (A.getHandle $ _param_in v) (A.getHandle $ _param_grad v) 0.01 nil)
                        return $ v {_param_in = new_in}
                    else return v
    ST.put $ Just params

-- forward only. Must provide all the placeholders, setting the data to 'Just ...', and set label to 'Nothing'.
-- note that the batch size here can be different from that in the training phase.
trainForwardOnly :: (MonadIO m, MonadThrow m) => SymbolF -> M.HashMap String (Maybe ArrayF) -> TrainM m [ArrayF]
trainForwardOnly net dat =
     ST.get >>= \case
        Nothing -> throwM NetworkUninitialized
        Just params -> liftIO $ do
            shps <- inferShape net (M.map fromJust $ M.filter isJust dat)
            params <- flip M.traverseWithKey params $ \k p -> do
                let ishp = shps M.! k
                case M.lookup k dat of
                    Just (Just a) -> 
                        return $ p {_param_in = a}
                    Just Nothing  -> do
                        dummy <- zeros ishp
                        return $ p {_param_in = dummy}
                    Nothing -> do
                        (_, pshp) <- ndshape (_param_in p)
                        when (ishp /= pshp) (throwM $ MismatchedShape k)
                        return p
            exec <- bindParam net params False
            checked $ mxExecutorForward (E.getHandle exec) 0
            getOutputs exec

data Exc = NetworkUninitialized | MismatchedShape String
    deriving (Show, Typeable)
instance Exception Exc

main :: IO ()
main = do
  _  <- mxListAllOpNames
  net <- neural
  runResourceT $ train $ do 
    liftIO $ putStrLn $ "[Train] "
    ST.forM_ (range 5) $ step net trainingData  
    liftIO $ putStrLn $ "[Test] "
    result <- SR.toList_ $ flip SR.mapM testingData $ \(x, y) -> do 
        [y'] <- trainForwardOnly net (M.fromList [("x", Just x), ("y", Nothing)])
        return (y, y')
    liftIO $ print $ take 10 result

  where
    range :: Int -> [Int]
    range = enumFromTo 1
    step net dat ind = do
        liftIO $ putStrLn $ "iteration " ++ show ind
        SR.mapM_ (\(x, y) -> trainStep net $ M.fromList [("x", x), ("y", y)]) dat

-- foo = do
--     _  <- mxListAllOpNames
--     net <- neural
--     Just (x,y) <- runResourceT $ SR.head_ trainingData
--     params <- initParam net (M.fromList [("x",x),("y",y)])
--     params <- return $ M.adjust (\p -> p{_param_in = x}) "x" params
--     params <- return $ M.adjust (\p -> p{_param_in = y}) "y" params
--     print $ params M.! "b2"
--     exec <- bindParam net params True
--     checked $ mxExecutorForward (E.getHandle exec) 1
--     backward exec
--     print $ params M.! "b2"
--     putStrLn "<update>"
--     params <- flip M.traverseWithKey params $ \ k v -> do
--         if (k /= "x" && k /= "y")
--             then do
--                 new_in <- A.NDArray <$> A.sgd_update (A.getHandle $ _param_in v) (A.getHandle $ _param_grad v) 0.01 nil
--                 return $ v {_param_in = new_in}
--             else return v
--     putStrLn "<final>"
--     print $ params M.! "b2"

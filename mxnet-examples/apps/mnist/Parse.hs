module Parse where

import Streaming
import Data.Attoparsec.ByteString as AP
import Data.Attoparsec.Binary as AP
import Data.Attoparsec.ByteString.Streaming as APS
import qualified Data.ByteString.Streaming as BSS
import qualified Data.ByteString.Internal as BS
import qualified Data.ByteString as BS
import qualified Data.Vector.Storable as SV
import qualified Data.Vector as NV
-- import Control.Monad.Morph (lift)
import Control.Monad (MonadPlus(..))
import Control.Monad.IO.Class
import Control.Exception.Base
import Control.Monad.Trans.Resource (MonadResource(..), MonadThrow(..), runResourceT)
import Data.Typeable

type Image = SV.Vector Float
type Label = Int

data Header = HeaderImg Int Int Int
            | HeaderLbl Int

header :: AP.Parser Header
header = do
  mc <- AP.anyWord32be
  case mc of
    0x00000803 -> do 
      [d1,d2,d3] <- AP.count 3 AP.anyWord32be
      return $ HeaderImg (fromIntegral d1) (fromIntegral d2) (fromIntegral d3)
    0x00000801 -> do 
      d1 <- AP.anyWord32be
      return $ HeaderLbl (fromIntegral d1)
    _ -> fail "Header type not recognised"

image :: Int -> Int -> AP.Parser Image
image w h = do
  BS.PS fp ofs len <- AP.take (w*h)
  let vw = SV.unsafeFromForeignPtr fp ofs len
  return $ SV.map ((/255) . fromIntegral) vw

label :: AP.Parser Label
label = fromIntegral <$> AP.anyWord8

sourceImages :: MonadResource m => FilePath -> Stream (Of Image) m ()
sourceImages fp = do
  (result, rest)<- lift $ APS.parse header (BSS.readFile fp)
  case result of
    Left (HeaderImg _ w h) -> void $ APS.parsed (image w h) rest
    _ -> throwM NotImageFile

sourceLabels :: MonadResource m => FilePath -> Stream (Of Label) m ()
sourceLabels fp = do
  (result, rest)<- lift $ APS.parse header (BSS.readFile fp)
  case result of
    Left (HeaderLbl _) -> void $ APS.parsed label rest
    _ -> throwM NotImageFile

data Exc = NotImageFile | NotLabelFile
    deriving (Show, Typeable)
instance Exception Exc

-- sourceImages :: AP.Parser (NV.Vector Image)
-- sourceImages = do
--   h <- header
--   case h of 
--     HeaderImg n w h -> NV.fromList <$> count n (image w h) 
--     _ -> mzero

-- sourceLabels :: AP.Parser (NV.Vector Label)
-- sourceLabels = do
--   h <- header
--   case h of 
--     HeaderLbl n -> NV.fromList <$> count n label 
--     _ -> mzero

-- parseImages :: MonadResource m => FilePath -> m (NV.Vector Image)
-- parseImages fp = do
--   content <- liftIO $ BS.readFile fp
--   case AP.parse sourceImages content of
--     Done _ r -> return r
--     _ -> throwM NotImageFile

-- parseLabels :: MonadResource m => FilePath -> m (NV.Vector Label)
-- parseLabels fp = do
--   content <- liftIO $ BS.readFile fp
--   case AP.parse sourceLabels content of
--     Done _ r -> return r
--     _ -> throwM NotLabelFile

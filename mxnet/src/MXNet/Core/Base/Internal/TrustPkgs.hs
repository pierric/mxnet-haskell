-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.TrustPkgs
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Trust the thirty-party packages
-- - tuple-ops
--
{-# LANGUAGE Trustworthy #-}
module MXNet.Core.Base.Internal.TrustPkgs (
    module Data.Tuple.Ops
) where

import Data.Tuple.Ops

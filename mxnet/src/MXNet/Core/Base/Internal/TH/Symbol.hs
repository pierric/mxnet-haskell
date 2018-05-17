-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.TH.Symbol
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Functions about Symbol that generated by template haskell.
--
{-# OPTIONS_GHC -Wno-missing-signatures #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# OPTIONS_GHC -Wno-unused-local-binds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module MXNet.Core.Base.Internal.TH.Symbol where

import Data.Proxy
import Data.Maybe

import MXNet.Core.Base.HMap
import MXNet.Core.Base.Internal
import MXNet.Core.Base.Internal.TH (registerSymbolOps)
import MXNet.Core.NNVM.Internal (nnGetOpHandle, nnSymbolCompose)
import Prelude hiding (sin, sinh, cos, cosh, tan, tanh, min, max, round, floor,
                       abs, sum, sqrt, log, exp, flip, concat, repeat, reverse)

-- | Register symbol operators.
$(registerSymbolOps)

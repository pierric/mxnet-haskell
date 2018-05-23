-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.HMap
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Updatable heterogeneous map.
--
-- @
-- > let a = add @"a" (1 :: Int) nil
-- > a
-- [a = 1]
-- > let b = update @"a" (+1) a
-- > b
-- [a = 2]
-- > let c = add @"b" (Nothing :: Maybe Float) b
-- > c
-- [b = Nothing, a = 2]
-- > set @"b" (Just 128) c
-- [b = Just 128.0, a = 2]
-- @
--
{-# OPTIONS_GHC -Wno-unused-foralls #-}
{-# OPTIONS_GHC -Wno-unused-type-patterns #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

{-# LANGUAGE IncoherentInstances #-}
{-# LANGUAGE ConstraintKinds #-}

module MXNet.Core.Base.HMap
    ( -- * HMap type definition
      HMap
      -- * Type level constraints and operators
    , KV (..)
    , ShowKV (..)
    , MatchKVList (..)
      -- * Operations on HMap.
    , nil
    , add
    , add'
    , (.+.)
    , get
    , (.->.)
    , update
    , set
    , mergeTo
    , dump
    , QueryKV
    , query
    ) where

import           GHC.TypeLits
import           Data.List (intercalate)
import           Data.Monoid ((<>))
import           Data.Proxy (Proxy (..))
import           Data.Typeable (Typeable, typeOf)
import           Data.Constraint (Dict(..), (:-)(..), mapDict)
import           Data.Type.Equality ((:~:)(..), testEquality)
import           Type.Reflection (typeRep)
import           Unsafe.Coerce (unsafeCoerce)

data KV v = Symbol := v

infixr 6 :=

data KVList (kvs :: [KV *]) where
    Nil :: KVList '[]
    Cons :: v -> KVList kvs -> KVList (k ':= v ': kvs)

-- | If a KVList has a specified type of KV pair.
data IfHasKey = Yes Symbol | No
  deriving Typeable

-- | Find specified key-value type pair in KVList.
type family FindKV (k :: Symbol) v (kvs :: [KV *]) :: IfHasKey where
    FindKV k _ '[] = 'No
    FindKV k v (k ':= v ': kvs) = 'Yes k
    FindKV k1 v1 (k2 ':= v2 ': kvs) = FindKV k1 v1 kvs

-- | HMap definition.
newtype HMap (kvs :: [KV *]) = HMap { getKVList :: KVList kvs }

-- | Constraint ensure 'HMap' must contain k-v pair.
class InDict (k :: Symbol) (v :: *) (kvs :: [KV *]) | k kvs -> v where
    get' :: HMap kvs -> v
    update' :: (v -> v) -> HMap kvs -> HMap kvs

instance {-# OVERLAPPING #-} InDict k v (k ':= v ': kvs) where
    get' (HMap (Cons v _)) = v
    {-# INLINE get' #-}
    update' f (HMap (Cons v kvs)) = HMap $ Cons (f v) kvs
    {-# INLINE update' #-}

instance (InDict k v kvs, 'Yes k ~ FindKV k v (k' ':= v' ': kvs)) => InDict k v (k' ':= v' ': kvs) where
    get' (HMap (Cons _ kvs)) =  get' @k (HMap kvs)
    {-# INLINE get' #-}
    update' f (HMap (Cons v kvs)) = HMap $ Cons v (getKVList $ update' @k f (HMap kvs))
    {-# INLINE update' #-}

-- | Create an empty HMap.
nil :: HMap '[]
nil = HMap Nil

{-# INLINE nil #-}

-- | Add a key-value pair into the HMap (via TypeApplications).
add :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
add v (HMap kvs) = HMap (Cons v kvs)

{-# INLINE add #-}

-- | Add a key-value pair into the HMap (via TypeApplications).
--
-- FIXME should have a @'No ~ FindKV k v kvs@ constraint here.
add' :: forall k v kvs. Proxy k -> v -> HMap kvs -> HMap (k ':= v ': kvs)
add' _ v (HMap kvs) = HMap (Cons v kvs)

{-# INLINE add' #-}


-- | Infix version of @add@.
(.+.) :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
(.+.) = add

infixr 8 .+.

{-# INLINE (.+.) #-}

-- | Get the value of an existing key.
get :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
get = get' @k

{-# INLINE get #-}

-- | Infix version of @get@.
(.->.) :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
(.->.) = get @k

infix 7 .->.

{-# INLINE (.->.) #-}

-- | Update the value of an existing key.
update :: forall (k :: Symbol) v kvs. InDict k v kvs => (v -> v) -> HMap kvs -> HMap kvs
update = update' @k

{-# INLINE update #-}

-- | Set the value of an existing key.
set :: forall k v kvs. InDict k v kvs => v -> HMap kvs -> HMap kvs
set v = update' @k (const v)

{-# INLINE set #-}

-- | Merge the first KVList into the second one.
class MatchKVList (kvs1 :: [KV *]) (kvs2 :: [KV *]) where
    -- | Update all values in the first HMap into the second KVList.
    mergeTo' :: HMap kvs1 -> HMap kvs2 -> HMap kvs2

instance MatchKVList ('[]) (kvs2) where
    mergeTo' _ m2 = m2

instance (MatchKVList kvs1 kvs2, InDict k v kvs2) => MatchKVList (k ':= v ': kvs1) kvs2 where
    mergeTo' (HMap (Cons v kvs)) m2 = mergeTo' (HMap kvs) (set @k v m2)

-- | Update all values in the first HMap into the second KVList.
mergeTo :: forall (kvs1 :: [KV *]) (kvs2 :: [KV *]). MatchKVList kvs1 kvs2 => HMap kvs1 -> HMap kvs2 -> HMap kvs2
mergeTo = mergeTo'

class ShowKV (kvs :: [KV *]) where
    show' :: forall k v. KVList kvs -> [(String, String)]

instance ShowKV '[] where
    show' _ = []
    {-# INLINE show' #-}

instance (KnownSymbol k, Typeable v, Show v, ShowKV kvs) => ShowKV (k ':= v ': kvs) where
    show' (Cons v kvs') = showImpl v : show' kvs'
        where showImpl value = (symbolVal (Proxy :: Proxy k), if typeOf value == typeOf "" then (init . tail . show)  value else show value) -- special rule for string value.
    {-# INLINE show' #-}

instance ShowKV kvs => Show (HMap kvs) where
    show m = "[" <> (intercalate ", " . map (\(k, v) -> k <> " = " <> v) . show' . getKVList $ m) <> "]"
    {-# INLINE show #-}

-- | Dump key-value pair in HMap as [(k, v)].
dump :: forall kvs. ShowKV kvs => HMap kvs -> [(String, String)]
dump = show' . getKVList

{-# INLINE dump #-}

type QueryKV kvs = (FindKVAlwaysTypeable kvs, FindKVEntailsInDict kvs)

query :: forall (k :: Symbol) v (kvs :: [KV *]). (KnownSymbol k, Typeable v, FindKVAlwaysTypeable kvs, FindKVEntailsInDict kvs) 
      => HMap kvs -> Maybe v
query m = case findKVTypeable @kvs (Proxy :: Proxy k) (Proxy :: Proxy v) of
            Dict -> case typeRep @(FindKV k v kvs) `testEquality` typeRep @('Yes k) of
                      Just Refl -> case indictEvidence @kvs (Proxy :: Proxy k) (Proxy :: Proxy v) of Sub Dict -> Just (get' @k m)
                      Nothing -> Nothing

-- | Making an constraint by discruitinize the `FindKV k v (k1 ':= v1 ': kvs)` (the law of Excluded Middle). 
-- Note that an axiom `(FindKV k v (k1 ':= v1 ': kvs) ~ FindKV k v kvs)` is introduced by 'unsafeCoerce',
-- because it is not possible to impose the fact that `k /= k1 \/ v /= v1` as in the GHC's type world.
em :: forall k v k1 v1 kvs a. (KnownSymbol k, KnownSymbol k1, Typeable v, Typeable v1) 
   => Proxy k -> Proxy v -> Proxy k1 -> Proxy v1 -> Proxy kvs
   -> ((k ~ k1, v ~ v1) :- a) -> ((FindKV k v (k1 ':= v1 ': kvs) ~ FindKV k v kvs) :- a) -> Dict a
em _ _ _ _ _ f g = 
    case (typeRep @k `testEquality` typeRep @k1, typeRep @v `testEquality` typeRep @v1) of
      (Just Refl, Just Refl) -> case f of Sub Dict -> Dict
      _ -> let axiom = unsafeCoerce (Dict :: Dict ())
           in mapDict g axiom

-- | An inductive construction of the constraint: `Typeable (FindKV k v kvs)` for all possible 
-- `k`, `v` and `kvs`.
class FindKVAlwaysTypeable (kvs :: [KV *]) where
    findKVTypeable :: (KnownSymbol k, Typeable v) => Proxy k -> Proxy v -> Dict (Typeable (FindKV k v kvs))

instance FindKVAlwaysTypeable '[] where
    findKVTypeable _ _ = Dict

instance (KnownSymbol k1, Typeable v1, FindKVAlwaysTypeable kvs) => FindKVAlwaysTypeable (k1 ':= v1 ': kvs) where
    findKVTypeable pk pv = em pk pv (Proxy :: Proxy k1) (Proxy :: Proxy v1) (Proxy :: Proxy kvs) (Sub Dict) (Sub $ findKVTypeable @kvs pk pv)

-- | An inductive construction of the entailment: if `FindKV k v kvs ~ 'Yes k` then `InDict k v kvs`.
class FindKVEntailsInDict (kvs :: [KV *]) where
    indictEvidence :: (KnownSymbol k, Typeable v) => Proxy k -> Proxy v -> ((FindKV k v kvs ~ 'Yes k) :- InDict k v kvs)

instance (KnownSymbol k1, Typeable v1) => FindKVEntailsInDict (k1 ':= v1 ': '[]) where
    indictEvidence pk pv = Sub $ em pk pv (Proxy :: Proxy k1) (Proxy :: Proxy v1) (Proxy :: Proxy ('[] :: [KV *]))
                        (Sub Dict) undefined
instance (FindKVEntailsInDict kvs, KnownSymbol k1, Typeable v1) => FindKVEntailsInDict (k1 ':= v1 ': kvs) where
    indictEvidence pk pv = Sub $ em pk pv (Proxy :: Proxy k1) (Proxy :: Proxy v1) (Proxy :: Proxy kvs) 
                                    (Sub Dict) (Sub $ case indictEvidence @kvs pk pv of Sub Dict -> Dict)

------------------------------------------------------------------------
-- Here is another way of defining the function `HMap kvs -> Maybe v`.
-- It has an advantage that exact `v` can be infered from the context,
-- But it also has one disadvantage that an explicit context 
-- `GetIfHasKey (FindK k kvs) v kvs` must be given.
------------------------------------------------------------------------

-- -- | Get the value of key by scrutinizing the IfHasKey.
-- class GetIfHasKey (haskey :: IfHasKey) v (kvs :: [KV *]) where
--     getIfHasKey :: Proxy haskey -> HMap kvs -> Maybe v

-- -- the compiler can deduce the actual 'v', because of the fact: 
-- -- InDict has a functional dependency 'k, kvs -> v'.
-- instance InDict k v kvs => GetIfHasKey ('Yes k) v kvs where
--     getIfHasKey _ m = Just $ get' @k m

-- -- in case 'k' does not appear, leaving 'v' as a free variable.
-- instance GetIfHasKey 'No v kvs where
--     getIfHasKey _ _ = Nothing

-- -- | Find specified key pair in KVList.
-- -- Note that this differs from 'FindKV' by removing the type argument 'v'. Therefore
-- -- type checking the 'get_' can proceed to 'GetIfHasKey' part without knowing the 
-- -- actuall of 'v'. In case of the 'k' exists, 'v' can be then automatically
-- -- inferred when inspecting 'InDict'.
-- type family FindK (k :: Symbol) (kvs :: [KV *]) :: IfHasKey where
--     FindK k '[] = 'No
--     FindK k (k ':= v ': kvs) = 'Yes k
--     FindK k1 (k2 ':= v ': kvs) = FindK k1 kvs

-- -- | Get the value of key if existing
-- get_ :: forall (k :: Symbol) v (kvs :: [KV *]). GetIfHasKey (FindK k kvs) v kvs => HMap kvs -> Maybe v
-- get_ = getIfHasKey (Proxy :: Proxy (FindK k kvs))

------------------------------------------------------------------------
-- Some other trials (avoiding introducing an axiom by 'unsafeCoerce')
------------------------------------------------------------------------
-- type family KVTypeable (kvs :: [KV *]) :: Constraint where
--     KVTypeable '[] = ()
--     KVTypeable (k ':= v : kvs) = (KnownSymbol k, Typeable v, KVTypeable kvs)
-- type E k v kvs = ((FindKV k v kvs ~ Yes k) :- InDict k v kvs, InDict k v kvs :- (FindKV k v kvs ~ Yes k))
-- class Y (kvs :: [KV *]) where
--     y :: (KnownSymbol k, Typeable v) => Proxy k -> Proxy v -> E k v kvs
-- -- instance Y '[] where
-- --     y = Sub no
-- instance (Y kvs, KnownSymbol k1, Typeable v1,  KVTypeable kvs) => Y (k1 ':= v1 ': kvs) where
--     y pk pv = let pk1= Proxy :: Proxy k1
--                   pv1= Proxy :: Proxy v1
--                   ck = T.typeOf pk `testEquality` T.typeOf pk1
--                   cv = T.typeOf pv `testEquality` T.typeOf pv1
--               in case (ck, cv) of 
--                    (Just Refl, Just Refl) -> (Sub Dict, Sub Dict)
--                    _ -> (Sub $ case C.trans (lemmaC pk pv) (C.trans (fst $ y @kvs pk pv) (lemmaA pk pv)) of
--                                  Sub Dict -> Dict
--                         ,Sub Dict)
--       where
--         lemmaA :: forall k v. (KnownSymbol k, Typeable v) 
--                => Proxy k -> Proxy v
--                -> (FindKV k v (k1 ':= v1 ': kvs) ~ Yes k) :- (FindKV k v kvs ~ Yes k)
--         lemmaA pk pv = Sub $ case axiomFindKVAlwaysTypeable @k @v @kvs of 
--                                     Dict -> case typeRep @(FindKV k v kvs) `testEquality` typeRep @(Yes k) of
--                                               Just Refl -> Dict
--                                               Nothing -> error "either k1/=k or v1/=v should hold in lemmaA."
--         lemmaB :: forall k v. (KnownSymbol k, Typeable v)
--                => Proxy k -> Proxy v
--                -> (FindKV k v kvs ~ Yes k) :- (FindKV k v (k1 ':= v1 ': kvs) ~ Yes k)
--         lemmaB pk pv = Sub $ case axiomFindKVAlwaysTypeable @k @v @(k1 ':= v1 ': kvs) of 
--                                     Dict -> case typeRep @(FindKV k v (k1 ':= v1 ': kvs)) `testEquality` typeRep @(Yes k) of
--                                               Just Refl -> Dict
--                                               Nothing -> error "either k1/=k or v1/=v should hold in lemmaA."                             
--         lemmaC :: forall k v. (KnownSymbol k, Typeable v) 
--                 => Proxy k -> Proxy v
--                 -> (InDict k v kvs :- InDict k v (k1 ':= v1 ': kvs))
--         lemmaC pk pv = Sub $ case C.trans (lemmaB pk pv) (snd $ y @kvs pk pv) of 
--                                       Sub Dict -> Dict
--         lemmaD :: forall k v. (KnownSymbol k, Typeable v) 
--                 => Proxy k -> Proxy v
--                 -> (InDict k v (k1 ':= v1 ': kvs) :- InDict k v kvs)
--         lemmaD pk pv = Sub $ Dict

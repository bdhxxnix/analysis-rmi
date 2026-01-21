#pragma once

#if __has_include("tlx/container/btree_map.hpp")
#include "tlx/container/btree_map.hpp"

namespace stx {
template <typename... Args>
using btree_map = tlx::btree_map<Args...>;
} // namespace stx
#else
#include <map>

namespace stx {
template <typename Key, typename Data, typename Compare = std::less<Key>,
          typename Alloc = std::allocator<std::pair<const Key, Data>>>
using btree_map = std::map<Key, Data, Compare, Alloc>;
} // namespace stx
#endif

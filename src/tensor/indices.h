#pragma once

/**
 * Metaprogramming utils for the Tensor class
 */

// Represent a pack of elements
template <size_t... elements> struct Pack {
  template <typename... T> static size_t reduce(T... args) {
    return ((elements * args) + ...);
  }
  template <typename... U> struct merge;
  template <size_t... newElements> struct merge<Pack<newElements...>> {
    using type = Pack<elements..., newElements...>;
  };

  template <size_t next> struct append {
    using type = Pack<elements..., next>;
  };

  template <template <typename, size_t...> typename C, typename T>
  using extract = C<T, elements...>;
};

// I only need the specialization...
template <size_t next, typename P> struct PackAppend;
// Append an element to a Pack
template <size_t next, size_t... elements>
struct PackAppend<next, Pack<elements...>> {
  using type = Pack<elements..., next>;
};

template <typename P, size_t first, size_t... elements> struct PopBackInternal {
  using tmp = typename P::append<first>::type;
  static constexpr size_t value = PopBackInternal<tmp, elements...>::value;
  using type = PopBackInternal<tmp, elements...>::type;
};

template <typename P, size_t first> struct PopBackInternal<P, first> {
  static constexpr size_t value = first;
  using type = P;
};

template <typename P> struct PopBack;
// Remove the last element of a Pack
template <size_t... elements>
requires(sizeof...(elements) >= 1) struct PopBack<Pack<elements...>> {
  static constexpr size_t value = PopBackInternal<Pack<>, elements...>::value;
  using type = PopBackInternal<Pack<>, elements...>::type;
};

template <typename P> struct PopFront;
// Remove the first element of a Pack
template <size_t first, size_t... elements>
struct PopFront<Pack<first, elements...>> {
  static constexpr size_t value = first;
  using type = Pack<elements...>;
};

template <typename P, size_t cache, size_t first, size_t... elements>
struct CumulativeProductInternal {
  using tmp = PackAppend<cache * first, P>::type;
  using type = CumulativeProductInternal<tmp, cache * first, elements...>::type;
};

// Base case
template <typename P, size_t cache, size_t first>
struct CumulativeProductInternal<P, cache, first> {
  using type = P;
};

// returns Pack<1, 1*indices[0], 1*indices[0]*indices[1], ...
template <size_t... indices> struct CumulativeProduct {
  using type = CumulativeProductInternal<Pack<1>, 1, indices...>::type;
};
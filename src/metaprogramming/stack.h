#pragma once

template<size_t first, size_t... others>
struct PopFrontInternal;

template<size_t... others>
using PopFrontInternalT = typename PopFrontInternal<others...>::Type;

template<size_t... others>
constexpr size_t PopFrontInternalV = PopFrontInternal<others...>::value;

template<typename S1, typename S2>
struct MergeStacks;

template<typename S1, typename S2>
using MergeStacksT = typename MergeStacks<S1, S2>::Type;

template<size_t...elements>
struct Stack{
    template<size_t new_el>
    struct PushFront{
        using Type = Stack<new_el, elements...>;
    };
    template<size_t new_el>
    using PushFrontT = typename PushFront<new_el>::Type;

    template<size_t new_el>
    struct PushBack{
        using Type = Stack<elements..., new_el>;
    };
    template<size_t new_el>
    using PushBackT = typename PushBack<new_el>::Type;

    struct PopFront{
        using Type = PopFrontInternalT<elements...>;
        static constexpr size_t value = PopFrontInternalV<elements...>;
    };
    using PopFrontT = typename PopFront::Type;
    static constexpr size_t PopFrontV = PopFront::value;
};

template<typename S, size_t x>
struct CountStack{
    using reducedStack = typename S::PopFrontT;
    static constexpr size_t value = ((S::PopFrontV == x) ? 1 : 0) + CountStack<reducedStack, x>::value;
};

template<size_t x>
struct CountStack<Stack<>, x>{
    static constexpr size_t value = 0;
};


template<typename S1, typename S2>
struct MergeStacks{
    using tmp = typename S2::PopFrontT;
    constexpr static size_t tmp_val = S2::PopFrontV; 
    using Type = MergeStacksT<typename S1::PushBackT<tmp_val>, tmp>;
};

template<typename S1>
struct MergeStacks<S1, Stack<>>{
    using Type = S1;
};


template<size_t first, size_t... others>
struct PopFrontInternal{
    using Type = Stack<others...>;
    static constexpr size_t value = first;
};
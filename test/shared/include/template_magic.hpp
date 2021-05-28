#pragma once

template <int...>
struct index_list {};

template <int N_count, typename indices, typename... T_list>
struct gen_indices_;

template <int N_count, int... indices, typename T_0, typename... T_rest>
struct gen_indices_<N_count, index_list<indices...>, T_0, T_rest...> {
  typedef typename gen_indices_<N_count + 1, index_list<indices..., N_count>,
                                T_rest...>::type type;
};

template <int N_count, int... indices>
struct gen_indices_<N_count, index_list<indices...>> {
  typedef index_list<indices...> type;
};

template <typename... T_list>
struct gen_indices : gen_indices_<0, index_list<>, T_list...> {};
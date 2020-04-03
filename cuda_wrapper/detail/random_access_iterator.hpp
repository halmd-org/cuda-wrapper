/*
 * Copyright (C) 2020       Jaslo Ziska
 * Copyright (C) 2010, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP
#define CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP

#include <type_traits>
#include <iterator>

namespace cuda {
namespace detail {

/**
 * Random access iterator with given base iterator and category.
 * Inspired by __gnu_cxx::__normal_iterator from stl_iterator.h
 */
template <typename Iterator, class Category>
class random_access_iterator
{
protected:
    Iterator current_;

public:
    typedef Category iterator_category;
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type
        difference_type;
    typedef typename std::iterator_traits<Iterator>::reference reference;
    typedef typename std::iterator_traits<Iterator>::pointer pointer;

    random_access_iterator() : current_(Iterator()) {}

    explicit random_access_iterator(const Iterator& i) : current_(i) {}

    template <typename OtherIterator>
    random_access_iterator(const random_access_iterator<
        OtherIterator,
        typename std::enable_if<
            std::is_convertible<OtherIterator, Iterator>::value,
            Category
    >::type>& other) : current_(other.base()) {}

    const Iterator& base() const { return current_; }

    reference operator*() const { return* current_; }
    pointer operator->() const { return current_; }
    reference operator[](const difference_type& n) const
    {
        return current_[n];
    }

    random_access_iterator& operator++()
    {
        ++current_;
        return* this;
    }
    random_access_iterator operator++(int)
    {
        return random_access_iterator(current_++);
    }

    random_access_iterator& operator--()
    {
        --current_;
        return* this;
    }
    random_access_iterator operator--(int)
    {
        return random_access_iterator(current_--);
    }

    random_access_iterator& operator+=(const difference_type& n)
    {
        current_ += n;
        return* this;
    }

    random_access_iterator operator+(const difference_type& n) const
    {
        return random_access_iterator(current_ + n);
    }

    random_access_iterator& operator-=(const difference_type& n)
    {
        current_ -= n;
        return* this;
    }

    random_access_iterator operator-(const difference_type& n) const
    {
        return random_access_iterator(current_ - n);
    }
};

template <typename IteratorL, typename IteratorR, class Category>
inline bool operator==(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() == rhs.base();
}
template <typename IteratorL, typename IteratorR, class Category>
inline bool operator!=(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() != rhs.base();
}
template <typename IteratorL, typename IteratorR, class Category>
inline bool operator<(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() < rhs.base();
}
template <typename IteratorL, typename IteratorR, class Category>
inline bool operator>(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() > rhs.base();
}
template <typename IteratorL, typename IteratorR, class Category>
inline bool operator<=(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() <= rhs.base();
}
template <typename IteratorL, typename IteratorR, class Category>
inline bool operator>=(const random_access_iterator<IteratorL, Category>& lhs,
    const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() >= rhs.base();
}

template<typename IteratorL, typename IteratorR, class Category>
inline typename random_access_iterator<IteratorL, Category>::difference_type
operator-(const random_access_iterator<IteratorL, Category>& lhs,
          const random_access_iterator<IteratorR, Category>& rhs)
{
    return lhs.base() - rhs.base();
}
template<typename Iterator, class Category>
inline typename random_access_iterator<Iterator, Category>::difference_type
operator-(const random_access_iterator<Iterator, Category>& lhs,
          const random_access_iterator<Iterator, Category>& rhs)
{
    return lhs.base() - rhs.base();
}
template <typename Iterator, class Category>
inline random_access_iterator<Iterator, Category>
operator+(typename random_access_iterator<Iterator, Category>::difference_type
    n, const random_access_iterator<Iterator, Category>& i)
{
    return random_access_iterator<Iterator, Category>(i.base() + n);
}

} // namespace detail
} // namespace cuda

#endif /* CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP */

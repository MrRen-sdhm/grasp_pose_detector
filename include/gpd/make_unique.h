//
// Created by hustac on 7/30/19.
//

#ifndef GPD_MAKE_UNIQUE_H
#define GPD_MAKE_UNIQUE_H

#include <memory>

namespace std {
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args &&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

#endif //GPD_MAKE_UNIQUE_H

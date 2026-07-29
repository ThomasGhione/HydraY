#pragma once

#include <cstddef>
#include <string_view>

namespace ascii {

[[nodiscard]] constexpr char toLower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] constexpr char toUpper(char c) noexcept {
    return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
}

[[nodiscard]] constexpr bool isSpace(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

[[nodiscard]] constexpr bool isDigit(char c) noexcept {
    return c >= '0' && c <= '9';
}

[[nodiscard]] constexpr bool iequals(std::string_view lhs, std::string_view rhs) noexcept {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (toLower(lhs[i]) != toLower(rhs[i])) return false;
    }
    return true;
}

} // namespace ascii

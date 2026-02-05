#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Base ID type: supports up to 65,536 different base types
using base_id = uint16_t;

// Reserved base IDs for standard bases (ASCII-compatible for backward compatibility)
constexpr base_id BASE_ID_A = 'A';
constexpr base_id BASE_ID_C = 'C';
constexpr base_id BASE_ID_G = 'G';
constexpr base_id BASE_ID_U = 'U';
constexpr base_id BASE_ID_a = 'a';
constexpr base_id BASE_ID_c = 'c';
constexpr base_id BASE_ID_g = 'g';
constexpr base_id BASE_ID_u = 'u';

// Invalid base ID
constexpr base_id BASE_ID_INVALID = 0xFFFF;

class BaseEncoding {
public:
    BaseEncoding() : next_id_(256) {
        // Initialize standard ASCII bases (0-255 are reserved for ASCII)
        for (uint16_t i = 0; i < 256; ++i) {
            char32_t ch = static_cast<char32_t>(i);
            char_to_id_[ch] = static_cast<base_id>(i);
            if (id_to_char_.size() <= i) {
                id_to_char_.resize(i + 1);
            }
            id_to_char_[i] = ch;
        }
        // Initialize canonical map: by default, each base maps to itself
        canonical_map_.resize(256);
        for (base_id i = 0; i < 256; ++i) {
            canonical_map_[i] = i;
        }
        // Set standard lowercase -> uppercase canonical mappings
        canonical_map_['a'] = 'A';
        canonical_map_['c'] = 'C';
        canonical_map_['g'] = 'G';
        canonical_map_['u'] = 'U';
        canonical_map_['t'] = 'U';
        canonical_map_['T'] = 'U';

        // Initialize origin map and pairedwith map for standard bases
        origin_map_.resize(256, BASE_ID_INVALID);
        pairedwith_map_.resize(256);
        origin_map_['A'] = origin_map_['a'] = 'A';
        origin_map_['C'] = origin_map_['c'] = 'C';
        origin_map_['G'] = origin_map_['g'] = 'G';
        origin_map_['U'] = origin_map_['u'] = 'U';
        origin_map_['T'] = origin_map_['t'] = 'U';
        pairedwith_map_['A'] = pairedwith_map_['a'] = "U";
        pairedwith_map_['C'] = pairedwith_map_['c'] = "G";
        pairedwith_map_['G'] = pairedwith_map_['g'] = "CU";
        pairedwith_map_['U'] = pairedwith_map_['u'] = "AG";
        pairedwith_map_['T'] = pairedwith_map_['t'] = "A";
    }

    // Register a new base character with its canonical (parent) base
    // Returns the assigned base_id
    base_id register_base(char32_t ch, base_id canonical) {
        auto it = char_to_id_.find(ch);
        if (it != char_to_id_.end()) {
            // Already registered, update canonical mapping
            canonical_map_[it->second] = canonical;
            return it->second;
        }

        // Assign new ID
        base_id new_id = next_id_++;
        char_to_id_[ch] = new_id;

        // Expand vectors as needed
        if (id_to_char_.size() <= new_id) {
            id_to_char_.resize(new_id + 1);
        }
        id_to_char_[new_id] = ch;

        if (canonical_map_.size() <= new_id) {
            canonical_map_.resize(new_id + 1);
        }
        canonical_map_[new_id] = canonical;

        return new_id;
    }

    // Register a base using a UTF-8 code string (single character)
    base_id register_base(const std::string& utf8_char, base_id canonical) {
        char32_t ch = decode_first_char(utf8_char);
        return register_base(ch, canonical);
    }

    // Get the canonical (parent) base ID for a given base ID
    base_id get_canonical(base_id id) const {
        if (id < canonical_map_.size()) {
            return canonical_map_[id];
        }
        return id; // Return itself if not found
    }

    // Get the origin (parent) base ID for a given base ID
    base_id get_origin(base_id id) const {
        if (id < origin_map_.size() && origin_map_[id] != BASE_ID_INVALID) {
            return origin_map_[id];
        }
        return get_canonical(id);  // Fallback to canonical
    }

    // Get the pairedwith string for a given base ID
    const std::string& get_pairedwith(base_id id) const {
        static const std::string empty;
        if (id < pairedwith_map_.size()) {
            return pairedwith_map_[id];
        }
        return empty;
    }

    // Set the origin for a base ID
    void set_origin(base_id id, base_id origin) {
        if (origin_map_.size() <= id) {
            origin_map_.resize(id + 1, BASE_ID_INVALID);
        }
        origin_map_[id] = origin;
    }

    // Set the pairedwith string for a base ID
    void set_pairedwith(base_id id, const std::string& pairedwith) {
        if (pairedwith_map_.size() <= id) {
            pairedwith_map_.resize(id + 1);
        }
        pairedwith_map_[id] = pairedwith;
    }

    // Check if two bases can pair (based on pairedwith information)
    bool can_pair(base_id id1, base_id id2) const {
        // Get origin for modified bases to check against pairedwith
        base_id origin1 = get_origin(id1);
        base_id origin2 = get_origin(id2);

        // Check if id2's origin is in id1's pairedwith
        char c2 = static_cast<char>(toupper(origin2 < 256 ? origin2 : 0));
        const auto& pw1 = get_pairedwith(id1);
        if (!pw1.empty() && c2 != 0 && pw1.find(c2) != std::string::npos) {
            return true;
        }

        // Check reverse: if id1's origin is in id2's pairedwith
        char c1 = static_cast<char>(toupper(origin1 < 256 ? origin1 : 0));
        const auto& pw2 = get_pairedwith(id2);
        if (!pw2.empty() && c1 != 0 && pw2.find(c1) != std::string::npos) {
            return true;
        }

        return false;
    }

    // Get the base_id for a char32_t character
    base_id get_id(char32_t ch) const {
        auto it = char_to_id_.find(ch);
        if (it != char_to_id_.end()) {
            return it->second;
        }
        return BASE_ID_INVALID;
    }

    // Get the base_id for a single-byte ASCII character
    base_id get_id(char ch) const {
        return static_cast<base_id>(static_cast<unsigned char>(ch));
    }

    // Get the char32_t for a base_id
    char32_t get_char(base_id id) const {
        if (id < id_to_char_.size()) {
            return id_to_char_[id];
        }
        return 0;
    }

    // Encode a UTF-8 string sequence into a vector of base_ids
    std::vector<base_id> encode(const std::string& utf8_seq) const {
        std::vector<base_id> result;
        result.reserve(utf8_seq.size()); // May be smaller for multibyte chars

        size_t i = 0;
        while (i < utf8_seq.size()) {
            char32_t ch;
            size_t bytes = decode_utf8_char(utf8_seq, i, ch);
            if (bytes == 0) {
                // Invalid UTF-8, treat as single byte
                ch = static_cast<unsigned char>(utf8_seq[i]);
                bytes = 1;
            }

            base_id id = get_id(ch);
            if (id == BASE_ID_INVALID) {
                // Unknown character, use ASCII value or 0
                id = (ch < 256) ? static_cast<base_id>(ch) : 0;
            }
            result.push_back(id);
            i += bytes;
        }

        return result;
    }

    // Get unique base IDs from a sequence
    std::unordered_set<base_id> get_unique_bases(const std::vector<base_id>& seq_ids) const {
        return std::unordered_set<base_id>(seq_ids.begin(), seq_ids.end());
    }

    // Check if a base_id is a standard ASCII base
    static bool is_ascii_base(base_id id) {
        return id < 256;
    }

    // Get lowercase version of a base_id (for standard bases only)
    static base_id to_lower(base_id id) {
        if (id >= 'A' && id <= 'Z') {
            return id + ('a' - 'A');
        }
        return id;
    }

private:
    std::unordered_map<char32_t, base_id> char_to_id_;
    std::vector<char32_t> id_to_char_;
    std::vector<base_id> canonical_map_;  // Maps each base_id to its canonical (parent) base
    std::vector<base_id> origin_map_;     // Maps each base_id to its origin (parent) base
    std::vector<std::string> pairedwith_map_;  // Maps each base_id to its pairing partners
    base_id next_id_;  // Next available ID for new bases

    // Decode a single UTF-8 character from a string starting at position pos
    // Returns the number of bytes consumed (0 if invalid)
    static size_t decode_utf8_char(const std::string& s, size_t pos, char32_t& out) {
        if (pos >= s.size()) {
            return 0;
        }

        unsigned char c = static_cast<unsigned char>(s[pos]);

        if ((c & 0x80) == 0) {
            // Single byte (ASCII)
            out = c;
            return 1;
        } else if ((c & 0xE0) == 0xC0) {
            // Two bytes
            if (pos + 1 >= s.size()) return 0;
            unsigned char c2 = static_cast<unsigned char>(s[pos + 1]);
            if ((c2 & 0xC0) != 0x80) return 0;
            out = ((c & 0x1F) << 6) | (c2 & 0x3F);
            return 2;
        } else if ((c & 0xF0) == 0xE0) {
            // Three bytes
            if (pos + 2 >= s.size()) return 0;
            unsigned char c2 = static_cast<unsigned char>(s[pos + 1]);
            unsigned char c3 = static_cast<unsigned char>(s[pos + 2]);
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return 0;
            out = ((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            return 3;
        } else if ((c & 0xF8) == 0xF0) {
            // Four bytes
            if (pos + 3 >= s.size()) return 0;
            unsigned char c2 = static_cast<unsigned char>(s[pos + 1]);
            unsigned char c3 = static_cast<unsigned char>(s[pos + 2]);
            unsigned char c4 = static_cast<unsigned char>(s[pos + 3]);
            if ((c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80 || (c4 & 0xC0) != 0x80) return 0;
            out = ((c & 0x07) << 18) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 6) | (c4 & 0x3F);
            return 4;
        }

        return 0; // Invalid UTF-8
    }

    // Decode the first character from a UTF-8 string
    static char32_t decode_first_char(const std::string& s) {
        char32_t ch = 0;
        decode_utf8_char(s, 0, ch);
        return ch;
    }
};

// Global encoding instance for shared use
inline BaseEncoding& get_global_encoding() {
    static BaseEncoding instance;
    return instance;
}

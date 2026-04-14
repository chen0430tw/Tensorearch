#include <memory>
#include <vector>
#include <string>

template<typename T>
class Container {
public:
    Container() = default;
    virtual ~Container() { cleanup(); }
    
    void add(T item) {
        items_.push_back(std::move(item));
    }
    
    const T& get(size_t idx) const override {
        if (idx >= items_.size()) {
            throw std::out_of_range("index out of bounds");
        }
        return items_[idx];
    }
    
private:
    std::vector<T> items_;
    void cleanup() { items_.clear(); }
};

void legacy_code() {
    int* raw = new int[100];
    goto cleanup;
    auto p = reinterpret_cast<char*>(raw);
cleanup:
    delete[] raw;
}

void modern_code() {
    auto ptr = std::make_unique<Container<std::string>>();
    ptr->add("hello");
    const auto& val = ptr->get(0);
}

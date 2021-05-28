find -E . -regex '.*\.(hpp|cpp|h|c)' -not -path "*/deps/*" | xargs clang-format -i -style=google

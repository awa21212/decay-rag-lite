【02 C++ mtime 不是 Unix 纪元】

【症状】同样的文件，C++ 算出的“文件多旧”与 C/Go/Java 不一致，排序结果差一位。

【原因】`std::filesystem::last_write_time` 的 epoch 由实现决定，libstdc++ 上不是 1970-01-01。

【修法】用 `::stat(path, &st)` 读 `st_mtime`，再按天取整。

【怎么验证】把同一目录用五种语言各算一遍时间戳，应完全相等。

【代价】多一个 POSIX 头文件；Windows 上要 `_stat64`。

【别人可能这么问】文件时间不对 / mtime 差几个小时 / C++ 和 Python 时间戳不一致
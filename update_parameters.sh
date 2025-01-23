#!/bin/bash

# 从 example.py 文件中读取 MY_VAR 的值
MY_VAR=$(grep 'MY_VAR =' parameters.py | cut -d '"' -f 2)

# 获取当前日期和时间
current_datetime=$(date '+%Y-%m-%dT%H:%M:%S')

# 将时间戳添加到 MY_VAR 的值后面
FOLDER_NAME="${current_datetime} | ${MY_VAR} | "

# 输出结果（可选）
echo "FOLDER_NAME: $FOLDER_NAME"

# 使用 awk 命令更新或添加 FOLDER_NAME 变量
awk -v new_value="$FOLDER_NAME" '
BEGIN { found = 0 }
{
    if ($0 ~ /^FOLDER_NAME =/) {
        print "FOLDER_NAME = \"" new_value "\""
        found = 1
    } else {
        print $0
    }
}
END {
    if (found == 0) {
        print "FOLDER_NAME = \"" new_value "\""
    }
}
' parameters.py > temp.py && mv temp.py parameters.py

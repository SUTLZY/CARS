#!/bin/bash

# 定义 Python 脚本路径
PYTHON_SCRIPT="driver.py"
FILE="parameters.py"

monitor_pids() {
    ps aux | grep '[p]ython driver.py' | grep -v 'conda run' | awk '{print $2}' | while read PID; do
        if ! grep -q "^$PID\$" ../pids.txt; then
            echo $PID >> ../pids.txt
            echo $PID >> "$tmpfile"
            echo "Started python driver.py with PID $PID"
        else
            echo "PID $PID is already recorded."
        fi
    done
}

# 启动 Python 脚本并获取进程 ID
start_script() {
    #python3 $PYTHON_SCRIPT &  # 在后台运行 Python 脚本
    conda run -n ray_lyn python $PYTHON_SCRIPT  &
    sleep 2;
    tmpfile=$(mktemp)
    monitor_pids
    PID=$(cat "$tmpfile")   # 获取进程 ID
    rm "$tmpfile"
    echo "Started $PYTHON_SCRIPT with PID: $PID"
}

# 定义函数来检查进程是否还在运行
is_running() {
    ps -p $1 > /dev/null 2>&1
}


# >>>>>>>>>>>脚本开始>>>>>>>>>>>脚本开始>>>>>>>>>>>脚本开始>>>>>>>>>>>
# >>>>>>>>>>>脚本开始>>>>>>>>>>>脚本开始>>>>>>>>>>>脚本开始>>>>>>>>>>>

if grep -q "LOAD_MODEL = False" "$FILE"; then
    echo "LOAD_MODEL 为 False, 首次启动"
    # 首次启动driver.py流程
    echo "[`date +%F\ %T`] driver.py is offline, 首次启动, try to start..." >> ./logs/check_catnipp_es.log;
    sleep 1;
    # 调用 update_example.sh 脚本
    ./update_parameters.sh
    sleep 1;
    # 启动脚本
    start_script
else
    echo "LOAD_MODEL 为 True, 再次开始"
     echo "[`date +%F\ %T`] driver.py is offline, 非首次启动, 开始训练..." >> ./logs/check_catnipp_es.log;
    sleep 1;
    # 启动脚本
    start_script
fi

# 检测driver.py的运行状态，及时重启
while [ 1 ] ; do
sleep 3
    # 检查进程状态
    if  is_running $PID; then   #如果driver.py的PID在运行
        echo "Process $PID is running..."
        if grep -q "LOAD_MODEL = False" "$FILE"; then
            echo "LOAD_MODEL 为 False 1"
            sed -i 's/LOAD_MODEL = False/LOAD_MODEL = True/' "$FILE"
            sleep 1;
            echo "LOAD_MODEL 已更改为 True 2"
        else
            echo "LOAD_MODEL 已是 True 3"
        fi
    else
        sleep 1;
        echo "Process $PID has stopped"
        echo "driver.py is not running, restarting..."
        sleep 30;
        # ray stop --force
        #sleep 9;
        echo "[`date +%F\ %T`] driver.py is offline, reload model, try to restart..." >> ./logs/check_catnipp_es.log;
        start_script  # 重新启动脚本
    fi
sleep 56
done

# https://blog.csdn.net/u013468614/article/details/115301776
# chmod +x update_example.sh
# chmod +x main_script.sh
#!/bin/bash
ps aux|grep evaluation.sh |grep -v grep | awk '{print $2}' | while read pid ;do  kill -9 $pid;done;


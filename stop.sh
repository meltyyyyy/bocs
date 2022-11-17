#!/bin/sh

ps aux | grep "python bocs" | grep -v grep | awk '{print "kill -9", $2 }' | sh

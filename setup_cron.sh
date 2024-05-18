#!/bin/bash

# Directory where main.py is located
SCRIPT_DIR="/home/auto/CODE/TRBD-null-pipeline/trbdv0"

# Adding a cron job to run main.py every day at 11:10 AM
(crontab -l 2>/dev/null; echo "10 11 * * * cd $SCRIPT_DIR && /home/auto/miniconda3/bin/python main.py") | crontab -

echo "Cron job set to run main.py every day at 11:10 AM."

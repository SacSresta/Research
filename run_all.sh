#!/bin/bash
python data_fetcher.py &
python categorical_main_date.py &
wait

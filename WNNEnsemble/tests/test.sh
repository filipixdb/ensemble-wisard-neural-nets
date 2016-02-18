#!/bin/bash
tests=(
   'python test_german_dataset.py 20 encoded_german.data')

for i in ${!tests[@]}; do
   echo ${tests[i]}
   eval ${tests[i]}
   a=$?
   if [ $a -ne 0 ]; then
      exit $a
   fi
   echo ''
done

#!/bin/bash

timestamp() {
	date +"%y%m%d-%H%M"
}
TIMESTAMP=$(timestamp)
echo "Start time:   " ${TIMESTAMP}

#check ob auf GPU mit Hilfe von $HOST ?
PWD=$VAULT/PhD/DeepLearning/UV-wire/
DATA=$PWD/Data/
FOLDERRUNS=$PWD/TrainingRuns/
CODEFOLDER=$HPC/UVWireRecon/

array=( $* )
for((i=0; i<$#; i++)) ; do
	case "${array[$i]}" in
		--test) TEST="true";;
		--prepare) PREPARE="true";;
		--resume) RESUME="true";;
		--model) PARENT=${array[$i+1]};;
		-m) PARENT=${array[$i+1]};;
		--single) SINGLE="true";;
		-s) SINGLE="true";;
	esac
done 

if [[ $TEST == "true" ]] ||  [[ $PREPARE == "true" ]] ; then
	RUN=Dummy/
else
	if [[ $RESUME=="true" ]] && [[ ! -z $FOLDERRUNS$PARENT ]]
	then
		if [ -d $FOLDERRUNS$PARENT ]
		then
			if [[ $SINGLE != "true" ]]
			then
				RUN=$PARENT/$TIMESTAMP/
			else
				RUN=$PARENT/
			fi
		else
			echo "Model Folder (${PARENT}) does not exist" ; exit 1
		fi
	else
		RUN=$TIMESTAMP/
	fi
fi

mkdir -p $FOLDERRUNS/$RUN

echo

if [[ $PREPARE == "true" ]] ; then
	echo "(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat"
else
	#echo "(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat"
	(python $CODEFOLDER/run_cnn.py --in $DATA --runs $FOLDERRUNS --out $RUN ${@:1}) | tee $FOLDERRUNS/$RUN/log.dat
	echo
	echo 'Start time:   ' ${TIMESTAMP}
fi


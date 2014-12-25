
# analysing following output:
# th-128.risoe.dk      job-exclusive
# to count all the free and busy nodes
rm pbsnodes.dat 2>/dev/null
pbsnodes -l all | cut -c 22-35 >>pbsnodes.dat
frees=`expr 0`
exclusives=`expr 0`
others=`expr 0`
for i in `cat pbsnodes.dat`
do
	# echo $i
	if [ $i == free ]
	then
		frees=`expr $frees + 1`
	elif [ $i == job-exclusive ]
	then
		exclusives=`expr $exclusives + 1`
	else
		let others++
	fi
done

check=`expr $frees + $exclusives + $others`
echo ""
echo "free nodes           : " $frees
echo "job-exclusives       : " $exclusives
echo "others               : " $others
echo "----------------------------"
echo "sum                  : " $check
echo ""

#### analysing qstat:
# 77880.th-000              ...at_4384_4.bat tjul                   0 W workq
# to count the busy and free nodes
rm qstat.dat 2>/dev/null
qstat | grep "th-"| cut -c 69-70 >>qstat.dat

#-----------------------------------------------------------------------------------------
# # grab also all the users:
# # 78617.th-000              NM80             niet            575:00:5 R workq
# qstat | grep "th-"| cut -c 44-49 >>qstat-users.dat

# for k in `cat qstat-users.dat`
# do
	
# done

# # nodes per user
# 78620.th-000.risoe.d niet     workq    NM80        28753     7   1    --  51200 R 577:0
#-----------------------------------------------------------------------------------------

Cs=`expr 0`
Es=`expr 0`
Hs=`expr 0`
Qs=`expr 0`
Rs=`expr 0`
Ts=`expr 0`
Ws=`expr 0`
Ss=`expr 0`
others=`expr 0`
for k in `cat qstat.dat`
do
	#echo $k
	if [ $k == C ]
	then
		let Cs++
	elif [ $k == E ]
	then
		let Es++
	elif [ $k == H ]
	then
		let Hs++
	elif [ $k == Q ]
	then
		let Qs++
	elif [ $k == R ]
	then
		let Rs++
	elif [ $k == T ]
	then
		let Ts++
	elif [ $k == W ]
	then
		let Ws++
	elif [ $k == S ]
	then
		let Ss++
	else
		let others++
	fi
done
echo ""
echo "C -  Job is completed after having run                                  : " $Cs
echo "E -  Job is exiting after having run.                                   : " $Es
echo "H -  Job is held.                                                       : " $Hs
echo "Q -  job is queued, eligible to run or routed.                          : " $Qs
echo "R -  job is running.                                                    : " $Rs
echo "T -  job is being moved to new location.                                : " $Ts
echo "W -  job is waiting for its execution time (-a option) to be reached.   : " $Ws
echo "S -  (Unicos only) job is suspend.                                      : " $Ss
echo ""
echo "others                                                                  : " $others
echo ""

# rm node-list.dat 2>/dev/null
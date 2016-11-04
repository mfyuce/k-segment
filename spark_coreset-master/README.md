# spark_coreset
Spark Framework for running coresets

# running instructions for windows
create environment variable:
https://blogs.msdn.microsoft.com/arsen/2016/02/09/resolving-spark-1-6-0-java-lang-nullpointerexception-not-found-value-sqlcontext-error-when-running-spark-shell-on-windows-10-64-bit/

to put files in run_k-segment_coreset:
./go_k_segment_coreset.sh

from run_k-segment_coreset folder:
	%SPARK_BIN%\spark-submit.cmd --master local[4] tree.py
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l de -threads 5 < newstest2014_de.txt > test_cost_de.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l en -threads 5 < newstest2014_en.txt > test_cost_en.txt

perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < test_cost_de.txt > test_de_2014.txt
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < test_cost_en.txt > test_en_2014.txt
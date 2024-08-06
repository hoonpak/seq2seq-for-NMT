/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l de -threads 5 < newstest2014_de.txt > new_test_cost_de.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l en -threads 5 < newstest2014_en.txt > new_test_cost_en.txt

sed -E 's/-+/-/g; s/-/ - /g; s/„/ \&quot; /g; s/“/ \&quot; /g' new_test_cost_de.txt > new_test_cost_de_.txt
sed -E 's/-+/-/g; s/-/ - /g; s/„/ \&quot; /g; s/“/ \&quot; /g' new_test_cost_en.txt > new_test_cost_en_.txt
echo "- finish"

sed -E 's/[[:space:]]+/ /g' new_test_cost_de_.txt > new_test_cost_de.txt
sed -E 's/[[:space:]]+/ /g' new_test_cost_en_.txt > new_test_cost_en.txt
echo "space finish"

# perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < new_test_cost_de.txt > new_test_de_2014.txt
# perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < new_test_cost_en.txt > new_test_en_2014.txt
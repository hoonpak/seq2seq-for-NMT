/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l de -threads 8 < /root/seq2seq-for-NMT/dataset/training/commoncrawl.de-en.de > new_training_de.txt
/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l de -threads 8 < /root/seq2seq-for-NMT/dataset/training/europarl-v7.de-en.de >> new_training_de.txt
/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l de -threads 8 < /root/seq2seq-for-NMT/dataset/training/news-commentary-v9.de-en.de >> new_training_de.txt
echo "de finish"
/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l en -threads 8 < /root/seq2seq-for-NMT/dataset/training/commoncrawl.de-en.en > new_training_en.txt
/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l en -threads 8 < /root/seq2seq-for-NMT/dataset/training/europarl-v7.de-en.en >> new_training_en.txt
/root/seq2seq-for-NMT/dataset/training/tokenizer.perl -l en -threads 8 < /root/seq2seq-for-NMT/dataset/training/news-commentary-v9.de-en.en >> new_training_en.txt
echo "en finish"

sed -E 's/-+/-/g; s/-/ - /g; s/„/ \&quot; /g; s/“/ \&quot; /g' new_training_de.txt > new_training_de_.txt
sed -E 's/-+/-/g; s/-/ - /g; s/„/ \&quot; /g; s/“/ \&quot; /g' new_training_en.txt > new_training_en_.txt
echo "- finish"

sed -E 's/[[:space:]]+/ /g; s/^[[:space:]]+//; s/[[:space:]]+$//' new_training_de_.txt > new_training_de.txt
sed -E 's/[[:space:]]+/ /g; s/^[[:space:]]+//; s/[[:space:]]+$//' new_training_en_.txt > new_training_en.txt
echo "space finish"

head -c 1M new_training_de.txt > test.txt

# perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < new_training_de.txt > new_training_de_.txt
# perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < new_training_en.txt > new_training_en_.txt
# echo "##AT## finish"

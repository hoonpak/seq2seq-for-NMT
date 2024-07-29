/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l de -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/commoncrawl.de-en.de > np_training_de.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l de -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/europarl-v7.de-en.de >> np_training_de.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l de -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/news-commentary-v9.de-en.de >> np_training_de.txt
echo "de finish"
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l en -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/commoncrawl.de-en.en > np_training_en.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l en -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/europarl-v7.de-en.en >> np_training_en.txt
/hdd1/user19/bag/4.RNN/dataset/training/tokenizer.perl -l en -threads 8 < /hdd1/user19/bag/4.RNN/dataset/training/news-commentary-v9.de-en.en >> np_training_en.txt
echo "en finish"
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < np_training_de.txt > np_training_de_.txt
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < np_training_en.txt > np_training_en_.txt
echo "##AT## finish"

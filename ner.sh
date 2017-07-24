dir="embedding"
if [ ! -d "$dir" ]; then
    mkdir "embedding"
fi

file="embedding/vectors.npy"
if [ -f "$file" ]; then
	echo "$file found."
else
	url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9WU93NEI1bGhmYmc"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
file="embedding/words.pl"
if [ -f "$file" ]; then
    echo "$file found."
else
    url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9SC1mRXpkbWhfUDA"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
python ner.py --word_dir embedding/words.pl --vector_dir embedding/vectors.npy --train_dir data/train_sample.txt --dev_dir data/dev_sample.txt --test_dir data/test_sample.txt --num_lstm_layer 2 --num_hidden_node 64 --dropout 0.5 --batch_size 50 --patience 3
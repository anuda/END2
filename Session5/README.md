# Sentiment CLassification on Stanford Dataset

## Data Augmentations
- Random Deletion
```Python
def random_deletion(words, p=0.3):
    words_list=words.copy()
    if len(words_list) ==1: # return if single word
        return words_list
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words_list))
    if len(remaining) == 0: # if not left, sample a random word
        return [random.choice(words_list)] 
    else:
        return remaining
```
- Random Swap
```Python
def random_swap(sentence, n=5): 
    length = range(len(sentence)) 
    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1] 
    return sentence
```

## DataSet Creation
- Create Fields:
  - Define the tokenizer, mention if the data is sequential in nature
  - batch_first : If set to true, will have the batch dimension first such as [1,1,28,28] First 1 is the batch size. Rest is size of the image
  - is_target: if this field is target variable
  - Stop_words can also be provided here
  - unknown and padding token can be explicity defined
```Python
Review = data.Field(sequential = True, tokenize='spacy', batch_first=True,lower=True,
                   include_lengths=True,stop_words=nltk_stopwords)
Label = data.LabelField(sequential = True,tokenize='spacy', is_target=True,
                        batch_first=True,stop_words=nltk_stopwords)
```
- Create Examples:
    - Create a example object from list of reviews and sentiments from field tuple. 
    - To access individual elements:example[0].review,example[0].label
```Python
train_example = [data.Example.fromlist([df_train.review[i],df_train.sentiment[i]], fields)
for i in range(df_train.shape[0])]
test_example = [data.Example.fromlist([df_test.review[i],df_test.sentiment[i]], fields)
for i in range(df_test.shape[0])]
valid_example = [data.Example.fromlist([df_dev.review[i],df_dev.sentiment[i]], fields)
for i in range(df_dev.shape[0])]
```
- Create Dataset
```Python
train_Dataset = data.Dataset(train_example, fields)
test_Dataset = data.Dataset(test_example, fields)
valid_Dataset = data.Dataset(valid_example,fields)
```
## Model Training Logs

    Train Loss: 1.301 | Train Acc: 59.88%
	 Val. Loss: 1.587 |  Val. Acc: 29.96% 

	Train Loss: 1.279 | Train Acc: 62.30%
	 Val. Loss: 1.588 |  Val. Acc: 30.00% 

	Train Loss: 1.264 | Train Acc: 63.75%
	 Val. Loss: 1.589 |  Val. Acc: 29.73% 

	Train Loss: 1.254 | Train Acc: 64.89%
	 Val. Loss: 1.589 |  Val. Acc: 29.73% 

	Train Loss: 1.247 | Train Acc: 65.65%
	 Val. Loss: 1.590 |  Val. Acc: 29.69% 

	Train Loss: 1.241 | Train Acc: 66.17%
	 Val. Loss: 1.591 |  Val. Acc: 29.64% 

	Train Loss: 1.236 | Train Acc: 66.73%
	 Val. Loss: 1.591 |  Val. Acc: 29.73% 

	Train Loss: 1.232 | Train Acc: 67.09%
	 Val. Loss: 1.591 |  Val. Acc: 29.91% 

	Train Loss: 1.228 | Train Acc: 67.50%
	 Val. Loss: 1.591 |  Val. Acc: 29.91% 

	Train Loss: 1.224 | Train Acc: 67.90%
	 Val. Loss: 1.592 |  Val. Acc: 29.60% 
## 25 Model Predictions:

```
Review: Effective but too-tepid biopic
Sentiment: 3
Model Predictions:tensor(1, device='cuda:0')
--------------------
Review: If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: Emerges as something rare , an issue movie that 's so honest and keenly observed that it does n't feel like one .
Sentiment: 5
Model Predictions:tensor(1, device='cuda:0')
--------------------
Review: The film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .
Sentiment: 3
Model Predictions:tensor(2, device='cuda:0')
--------------------
Review: Offers that rare combination of entertainment and education .
Sentiment: 5
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: Perhaps no picture ever made has more literally showed that the road to hell is paved with good intentions .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: Steers turns in a snappy screenplay that curls at the edges ; it 's so clever you want to hate it .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: But he somehow pulls it off .
Sentiment: 4
Model Predictions:tensor(2, device='cuda:0')
--------------------
Review: Take Care of My Cat offers a refreshingly different slice of Asian cinema .
Sentiment: 4
Model Predictions:tensor(2, device='cuda:0')
--------------------
Review: This is a film well worth seeing , talking and singing heads and all .
Sentiment: 5
Model Predictions:tensor(3, device='cuda:0')
--------------------
Review: What really surprises about Wisegirls is its low-key quality and genuine tenderness .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: ( Wendigo is ) why we go to the cinema : to be fed through the eye , the heart , the mind .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: One of the greatest family-oriented , fantasy-adventure movies ever .
Sentiment: 5
Model Predictions:tensor(3, device='cuda:0')
--------------------
Review: Ultimately , it ponders the reasons we need stories so much .
Sentiment: 3
Model Predictions:tensor(1, device='cuda:0')
--------------------
Review: An utterly compelling ` who wrote it ' in which the reputation of the most famous author who ever lived comes into question .
Sentiment: 4
Model Predictions:tensor(1, device='cuda:0')
--------------------
Review: Illuminating if overly talky documentary .
Sentiment: 3
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: A masterpiece four years in the making .
Sentiment: 5
Model Predictions:tensor(2, device='cuda:0')
--------------------
Review: The movie 's ripe , enrapturing beauty will tempt those willing to probe its inscrutable mysteries .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: Offers a breath of the fresh air of true sophistication .
Sentiment: 5
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: A thoughtful , provocative , insistently humanizing film .
Sentiment: 5
Model Predictions:tensor(3, device='cuda:0')
--------------------
Review: With a cast that includes some of the top actors working in independent film , Lovely & Amazing involves us because it is so incisive , so bleakly amusing about how we go about our lives .
Sentiment: 5
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: A disturbing and frighteningly evocative assembly of imagery and hypnotic music composed by Philip Glass .
Sentiment: 3
Model Predictions:tensor(1, device='cuda:0')
--------------------
Review: Not for everyone , but for those with whom it will connect , it 's a nice departure from standard moviegoing fare .
Sentiment: 4
Model Predictions:tensor(3, device='cuda:0')
--------------------
Review: Scores a few points for doing what it does with a dedicated and good-hearted professionalism .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
Review: Occasionally melodramatic , it 's also extremely effective .
Sentiment: 4
Model Predictions:tensor(0, device='cuda:0')
--------------------
```

#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=48, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):  #concatenate all french sentence tuples into one long tuple.
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  # The following code implements a monotone decoding
  # algorithm (one that doesn't permute the target phrases).
  # Hence all hypotheses in stacks[i] represent translations of 
  # the first i words of the input sentence. You should generalize
  # this so that they can represent translations of *any* i words.
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")   #lm_state = for the current partial translation, the most recent set of english words
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]): #i represents the number of words that have been translated so far from the beginning (consider changing this coz you can select any french phrase anywhere in the french sentence)
    #represents the partial translations of the first i words in the french sentence
    #change it ot ANY i words rather than the first i words
    for h in sorted(stack.itervalues(),key=lambda h: -h.logprob)[:opts.s]: # prune  
      for j in xrange(i+1,len(f)+1):  #need to generalize # pick the next chunk of the snetence #should come from anywhere in the sentence 
      #i is the next word of the rightmost last word that was translated, j is the length of the words that are translated starting from i.
        if f[i:j] in tm:  #inclusive i, excluding j 
          for phrase in tm[f[i:j]]:   
            logprob = h.logprob + phrase.logprob  
            lm_state = h.lm_state #current state of the language model for the hypotheses we are currently looking at
            for word in phrase.english.split():
              (lm_state, word_logprob) = lm.score(lm_state, word)   #returns new state, and probability of word acc to lm
              logprob += word_logprob
            logprob += lm.end(lm_state) if j == len(f) else 0.0 #special end probability for end of sentence #when we generalize, make sure that you add it only when the whole english sentence is translated and not when you reach the end of the french sentence
            new_hypothesis = hypothesis(logprob, lm_state, h, phrase) 
            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob: # second case is recombination
              stacks[j][lm_state] = new_hypothesis 
  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  def extract_english(h): 
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
  print(extract_english(winner))

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

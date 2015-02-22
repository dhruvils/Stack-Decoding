#!/usr/bin/env python
import optparse
import sys
import models
from math import log10
from collections import namedtuple, defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=48, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
optparser.add_option("-a", "--distance-cost", dest="alpha", default=0.3, type="float", help="Value for alpha (between 0 and 1) for reordering distance penalty (0 is harshest penalty and 1 is the most lenient penalty)")
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

  # future cost estimation: god help me
  cost = defaultdict(lambda: -sys.maxint)
  for length in xrange(1, len(f) + 1):
    for start in xrange(0, len(f) + 1 - length):
      end = start + length
      cost[f[start:end]] = -sys.maxint 
      if f[start:end] in tm:
        phrase = max(tm[f[start:end]], key=lambda l: l.logprob)
        cost[f[start:end]] = phrase.logprob

        lm_state = ()
        for word in phrase.english.split():
          (lm_state, word_logprob) = lm.score(lm_state, word)
          cost[f[start:end]] += word_logprob

      for i in xrange(start, end):
        if f[start:i] in cost and f[i:end] in cost:
          if(cost[f[start:i]] + cost[f[i:end]]) > cost[f[start:end]]:
            cost[f[start:end]] = cost[f[start:i]] + cost[f[i:end]]
      # sys.stderr.write("cost: %s f[start:end]: %s " %(cost[f[start:end]], f[start:end]))


  hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, bitvec, end_i, futurecost")   #lm_state = for the current partial translation, the most recent set of english words

  bit_vec = [False for _ in f]

  futurecost = 0

  initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, bit_vec, 0, futurecost)
  stacks = [{} for _ in f] + [{}]
  stacks[0][str(bit_vec)] = initial_hypothesis

  for i, stack in enumerate(stacks[:-1]): #i represents the number of words that have been translated so far from the beginning (consider changing this coz you can select any french phrase anywhere in the french sentence)
    #represents the partial translations of the first i words in the french sentence
    #change it ot ANY i words rather than the first i words
    for h in sorted(stack.itervalues(),key=lambda h: -(h.logprob + h.futurecost))[:opts.s]: # prune  
      # sys.stderr.write("futurecost: %s " %(h.futurecost))
      for win_length in xrange(1, len(f) + 1): #check all possible window lengths
        for j in xrange(0,len(f) + 1 - win_length): #range of start values represented by j
          bit_vec = h.bitvec[:]
          end = j + win_length  #ending points of french phrases to consider

          #future cost calculation!
          start_fc = -1
          end_fc = -1
          futurecost = 0
          for bit_index,bit in enumerate(bit_vec):
            # sys.stderr.write("bitvec: %s\n" %(bit_vec))
            if not bit:
              start_fc = bit_index if start_fc == -1 else start_fc
              end_fc = bit_index
            else:
              if start_fc != -1:
                futurecost += cost[f[start_fc:end_fc + 1]]
              start_fc = -1
          if start_fc != -1:
            futurecost += cost[f[start_fc:end_fc + 1]]
          # sys.stderr.write("futurecost: %s\n" %(futurecost))

          #actual logprob calculations!
          if f[j:end] in tm and True not in bit_vec[j:end]: 
            for bit_index in xrange(j, end):
              bit_vec[bit_index] = True

            for phrase in tm[f[j:end]]:   
              logprob = h.logprob + phrase.logprob  
              lm_state = h.lm_state #current state of the language model for the hypotheses we are currently looking at

              for word in phrase.english.split():
                (lm_state, word_logprob) = lm.score(lm_state, word)   #returns new state, and probability of word acc to lm
                logprob += word_logprob

              logprob += lm.end(lm_state) if False not in bit_vec else 0.0 #special end probability for end of sentence #when we generalize, make sure that you add it only when the whole english sentence is translated and not when you reach the end of the french sentence
              
              #re-ordering penalty!
              distance = abs(j - h.end_i - 1)

              #distance based penalty
              if distance > 6:
                logprob += log10(opts.alpha ** distance)

              new_hypothesis = hypothesis(logprob, lm_state, h, phrase, bit_vec, end - 1, futurecost) 
              # if lm_state not in stacks[i + win_length - repeat] or stacks[i + win_length - repeat][lm_state].logprob < logprob: # second case is recombination
              #   stacks[i + win_length - repeat][lm_state] = new_hypothesis
              if str(bit_vec) not in stacks[i + win_length] or stacks[i + win_length][str(bit_vec)].logprob < logprob: # second case is recombination
                stacks[i + win_length][str(bit_vec)] = new_hypothesis 
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

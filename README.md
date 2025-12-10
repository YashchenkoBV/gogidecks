# gogidecks
Genetic algorithm deck builder for Clash Royale

## Problem
Clash Royale is a real-time video game combining elements of collectible card games (CCG), tower defense and multiplayer online battle arena. The collecting cards and building your own deck is the defining part of the gameplay. In this project we address the problem of completing the deck for user given that they want to necessarily include some cards in it. Although there are many tools providing the evaluation of deck, none can suggest what cards to add to the deck the strenghten it based not only on the results of matches from the game.

Gogidecks suits only for building decks for regular 1v1 battle, consisting of 8 cards. Each card may be either a troop, a building or a spell. Each card costs defined amount of elixir (mana), and has its own unique values of attack, hp, and other characteristics (e.g. the unit may be on the ground or in the air). We don't account the level of cards.

![Best-Clash-Royale-arena-decks jpg](https://github.com/user-attachments/assets/29dca7b1-7926-4b95-bee3-b60a070c5dfc)

Pic. 1 The example of Clash Royale deck (8 cards), in given case it contains 6 troops, 1 building and 1 spell (in this order)

## Dataset
There are 121 cards in total (as for Dec 2025, excluding different types of tower troops). For the evaluation purposes, we plan to manually grade the best proposed decks on the first stage (we are both Clash Royale players with more than 6 years of experience) or use RoyaleAPI (aggregating the results of matches and win-rates for the decks) afterwards.

## Encoding scheme
Since each deck should contain exactly 8 cards, we plan to encode the deck as an eight-gene chromosome (one gene - one card, encoded by a number from [121]). Noteworthy, the arrangement of cards inside of the deck doesn't change it, so for the sake of consistency, we will sort the genes in chromosome in ascending order (by card ID).

We plan to implement mutation and crossover as follows: the probability of transition into the card assigned to the same class (e.g. spells, or tanks) is higher than the probability of transition into the card of different class (but the latter is strictly positive). We don't have the partition of the cards into those classes yet, but this part shouldn't take long (there are 8-10 classes expected).

## Fitness function
We don't have the exact fitness function yet - basically, its defining is the most hard and valuable part of the project. We think of experimenting with different ones during the implementation time, but some basic ideas could be found below:

$F = w_{atk} \cdot atk + w_{def} \cdot def + w_{syn} \cdot syn + w_{vers} \cdot vers + w_{mana} \cdot mana$,

where

$atk = w_0 \cdot tanks + w_1 \cdot win-condition + w_2 \cdot big-attacking-spells \cdot + w_3 \cdot small-attacking-spells$,
$def = w_4 \cdot anti-air-units + w_5 \cdot buildings + w_6 \cdot swarms + w_7 \cdot defense-spells + w_8 \cdot anti-tank-units$,
$syn = pair-from-synergy-table$,
$vers = pair-from-versatility-table$,
$mana = average-mana-cost$.

For implementation of this fitness function, we would need to obtain a number of tables describing the relations between different cards. We plan to do so by web-scrapping from DeckShop - another web-tool for Clash Royale decks.

The theoretic basis for this project was partly inspired by
Deck Building in Collectible Card Games using Genetic Algorithms: A Case Study of Legends of Code and Magic, 2021 IEEE Symposium Series on Computational Intelligence (SSCI) | 978-1-7281-9048-8/21/$31.00 ©2021 IEEE | DOI: 10.1109/SSCI50451.2021.9659984

# Implementation

## Step 1. Data collection and preprocessing

The first step was to collect data about all cards in the game. The issue 
here was that there are no open-source datasets storing up-to-date information about all Clash Royale cards. We even check Kaggle,
but the most recent partially relevant data was updated more than 2 years ago.
In order to retrieve names and cost (in mana) of all cards we used web scraping from [deckshop.pro](https://www.deckshop.pro/) --
a benchmark-like service for evaluating Clash Royale decks and providing game advice. We also used web scraping to obtain 
attack synergies and counter lists (what cards are countered by each card) from this website. This data, however, was not sufficient
to implement a sophisticated enough fitness function. It was necessary to, somehow, be able to access information about class properties
of every card, such as being a tank unit, air unit, anti-swarm unit, building unit, etc. Moreover, with as complicated balance system as
it is currently in place in the game, being related to some class could not be encoded as a binary property. For example, card
"miner" can be used in different contexts both as a tank and as an attack or defence spell! For this reason, and also because
there are no existing datasets that would satisfy our requirements, we classified all cards by hand, setting for each card a value
between 0 and 1 for 13 class-related parameters. We also introduced a win-condition parameter for each car, with values between 0 and 1.5, in
order to encourage usage of particular cards, that were strongly connected to only a small amount of classes. In the same time,
the attack synergy and counter data (the latter used to determine versatility of a deck) were written into index-based matrices, in which
(i,j)-th entry meant card i works good with card j, or card i counters card j. In total, with mana cost and general 
category (troop, spell, building, anti-building), each card has 17 parameters.

## Step 2. Trying linear regression to understand fitness weights

Our first thought was to design a linear fitness function on 17 parameters, each being a sum of corresponding parameters of deck cards.
For synergy score, we counted pairs of cards in deck, which are present in synergy matrix. For versatility score,
we, naturally, summed values in the 8 rows of counter matrix, corresponding to cards in the deck. We had, therefore, 17 weights of a linear
function to determine. The obvious proceeding was to choose linear regression. For targets, we decided to hand-pick 50 popular Clash Royale decks
of different class and winrate scores, based of official match-up statistic. We used the winrates as target values.


However, we were quickly disenchanted by the obtained results, as approximately half of produced weights were negative.
Even after forcing non-cost weights to be positive, the outcome still left a lot to be desired. Here is what we got:

```bash
Feature weights:
cost : -0.792343
tank : +3.462093
air : +0.140046
close-combat : +2.418437
far-combat : +0.000000
win-condition : +1.651011
big-atk-spell : +6.415656
small-atk-spell : +1.057580
def-spell : +3.258826
anti-air : +0.000000
building : +1.471002
spawn : +0.000000
swarm : +0.000000
anti-swarm : +0.000000
anti-tank : +3.090190
num_synergy_pairs : +0.878874
total_counters : +0.012338
```

It is immediately obvious that these weights are not reasonable. For one thing, they completely disregard very important features such as anti-air and ranged (far-combat) units. Moreover, the model assigns a larger weight to defence spells (which are useful, but not absolutely essential in actual gameplay) than to win-conditions (one of the most important parameters in our model).

There are two main reasons why this approach failed. First, a set of 50 decks is far too small to be representative enough for a reliable regression model. Hand-picking more decks was not an option, as it would require many hours of manual work. Second, the deck quality landscape we are trying to model is likely too complex to be accurately approximated by a purely linear function.

## Step 3.
Because described above approaches couldn't work well enough (we check on both RoyaleAPI-based dataset with winrates and DeckShop evaluation data), we implemented non-linear logic and added hard-constaints (such as maximal number of cards assigned to some class). Fitness function was reworked to: 1) reward synergy and counter potential (linear combination of the number of pair in the deck also presented in the synergy table and the sum of number of cards countered by each of the cards in the deck - exactly this formulation, since it accounts possible overlaps), 2) punish unsufficient (by default, < 1) summary values of features such as tank, win-condition, anti-air etc., each by fixed value, 3) punish summary elixir cost if it is out of range - linear over margin value. So normal values of fitness function are small and positive. After that, genetic algorithm started to produce meaningful (in our opinion and by DeckShop evaluation) decks. Since the evaluation of fitness function by win rates failed, we decide to accept a new one without additional testing (however, we tuned the weights value).

# Genetic algorithm details
In the final version, genetic algorithm uses class-defined mutations (for any card to mutate the probability of mutation into the card of the same class ("troop", "spell", "building", "anti-building" - troop attacking only buildings) is significantly - 0.8 over 0.2 - higher than mutation into the card of other class). Noteworthy, mutation rate controls the probability to mutate for each of the cards in the deck. We used basic tournament scheme for crossover (no preferences here, equal probabilities for each card, each crossover results in two children) and elitism (by default, 2 best chromosomes pass the generation). Obviously, we also applied non-redundancy constraint - a process resulting in the presence of more that one identical card in the same deck cannot happen. For the purpose of tuning the parameters of the genetic algorithm itself, we evaluated it on the benchmark (benchmark/ga_benchmark.py, measures k-size, population size, mutation rate) and set the best parameters by default.

# Use
To use the project, please install the requirements. Genetic algorithm could be launched by
```
python3 src/genalg_main.py [--force [card slugs divided by space]] [parameters affecting algorithm]
python3 src/genalg_main.py --force hog-rider mega-knight --pop-size 200
```

You can also check the fitness function value for any set of cards (even not of size 8):
```
python3 src/fitness.py mega-knight mighty-miner
```

For the convinience of checking counter and synergy data we wrote simple script, launched by:
```
python3 benchmark/check_npy.py data/[counter_matrix.npy or synergy_matrix.npy] [card-slug-1 card-slug-2]
python3 benchmark/check_npy.py data/counter_matrix.npy valkyrie skeleton-army
```

Benchmark could be launched with any genalg_main.py parameters, e.g.:
```
python3 benchmark/ga_benchmark.py --force minion-horde --crossover-rate 0.7
```

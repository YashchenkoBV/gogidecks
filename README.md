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

$atk = w_0 \cdot #tanks + w_1 \cdot #win_condition + w_2 \cdot #big_attacking_spells \cdot + w_3 \cdot #small_attacking_spells$,
$def = w_4 \cdot #anti_air_units + w_5 \cdot #buildings + w_6 \cdot #swarms + w_7 \cdot #defense_spells + w_8 \cdot #anti_tank_units$,
$syn = #pair_from_synergy_table$,
$vers = #pair_from_versatility_table$,
$mana = average_mana_cost$.

For implementation of this fitness function, we would need to obtain a number of tables describing the relations between different cards. We plan to do so by web-scrapping from DeckShop - another web-tool for Clash Royale decks.

The theoretic basis for this project was partly inspired by
Deck Building in Collectible Card Games using Genetic Algorithms: A Case Study of Legends of Code and Magic, 2021 IEEE Symposium Series on Computational Intelligence (SSCI) | 978-1-7281-9048-8/21/$31.00 ©2021 IEEE | DOI: 10.1109/SSCI50451.2021.9659984

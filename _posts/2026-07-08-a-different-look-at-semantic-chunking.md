---
layout: post
title: "A different look at semantic chunking"
date: 2026-07-08 00:00:00 -0000
categories:
    - semantic-text-tiling
tags: ["numpy", "sentence-transformers", "embeddings"]
---

# A different look at semantic chunking

The text-chunking subdomain of LLM-based text processing is full of wonderful projects that can slice and dice any text you give them. However, I noticed that something is amiss here, and that is the first point I want to make. In some very specific use cases, simply slicing a text by headings or paragraphs may not be enough, especially if your work requires the code to run cheaply. I mean the whole code, not only the text-chunking part. Sure, it's a walk in the park for a modern, frontier LLM to do whatever needs to be done with a text chunk. The other side of that coin is that using an LLM on suboptimal input may incur more costs than you're willing to pay; the outputs may be hard to reproduce; and it can be quite slow for longer texts. On top of that, API prices spike from time to time, the APIs themselves are unstable, and their performance varies between peak and non-peak hours. That is where I'm coming from with this (still in alpha!) library.

The second point: think about RAG systems. Maybe yours is not that good, and users complain that the search in your system returns some really useless results? You inspect the inputs before embedding them, and you see that very different topics often end up in the same chunk and get embedded together, which makes that chunk's meaning very blurry?

And the third point: maybe you're trying to apply a patching strategy to some textual data, but you notice that a single change request may relate to a bunch of text fragments dispersed across different pages?

This library is trying to address all those pain points! The code lives on GitHub at [github.com/mmalek06/text_change_detector](https://github.com/mmalek06/text_change_detector). In this post, though, I'll only describe briefly how to use it, and go into the technical details in later posts :)

## Enough talk, let me run it on a real statute

To make this concrete, I grabbed two public-domain US antitrust laws from Wikisource, the Sherman Act (1890) and the Clayton Act (1914), stripped the wiki markup, and split them into sentences with spaCy. That gives 99 sentences of dense legal prose across 34 sections. Then the whole first step is a single call:

```python
from text_change_detector import tile
from text_change_detector.shared.models import Segment

# one spaCy sentence per Segment, in document order
segments = [Segment(text=sentence, section=label) for label, sentence in prepared]

tiling = tile(segments)   # default embedder: Qwen3-Embedding-4B
```

(`tile` also takes a `.docx` or `.pdf` path directly, in which case it does the extraction for you. Here I already had plain text, so I fed it `Segment`s.)

`tile` cut those 99 sentences into 22 **semantic units** (a semantic unit is a short run of adjacent sentences about one thing), then grouped those units into 3 **communities**. A community, in this library, is simply a cluster of units that turned out to be about the same broad topic, pulled together on a similarity graph even when they sit far apart in the document. Here is that graph, one node per unit, coloured by community:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/antitrust-tiling-communities.png" /><br />

The nice thing is that the three communities (blue, orange and green in the picture above) line up with three real facets of antitrust law, even though the tiler knows nothing about law and never sees the section numbers.

**Community 0, the blue nodes - what is actually forbidden** (plus the definitions and the treble-damages remedy):

```text
Every contract, combination in the form of trust or otherwise, or conspiracy, in restraint of trade or commerce among the several States, or with foreign nations, is hereby declared to be illegal.
    -- Sherman Act §1

Every person who shall monopolize, or attempt to monopolize, or combine or conspire with any other person or persons, to monopolize any part of the trade or commerce among the several States, or with foreign nations, shall be deemed guilty of a misdemeanor.
    -- Sherman Act §2 (the tiler put §1, §2 and §3 into a single unit, because they say almost the same thing)

That it shall be unlawful for any person engaged in commerce, in the course of such commerce, either directly or indirectly to discriminate in price between different purchasers of commodities, which commodities are sold for use, consumption, or resale within the United States or any Territory thereof or the District of Columbia or any insular possession or other place under the jurisdiction of the United States, where the effect of such discrimination may be to substantially lessen competition or tend to create a monopoly in any line of commerce.
    -- Clayton Act §2

That no corporation engaged in commerce shall acquire, directly or indirectly, the whole or any part of the stock or other share capital of another corporation engaged also in commerce, where the effect of such acquisition may be to substantially lessen competition between the corporation whose stock is so acquired and the corporation making the acquisition, or to restrain such commerce in any section or community, or tend to create a monopoly of any line of commerce.
    -- Clayton Act §7
```

**Community 1, the orange nodes - court injunctions and who may sue:**

```text
The several circuit courts of the United States are hereby invested with jurisdiction to prevent and restrain violations of this act; and it shall be the duty of the several district attorneys of the United States, in their respective districts, under the direction of the Attorney-General, to institute proceedings in equity to prevent and restrain such violations.
    -- Sherman Act §4

No temporary restraining order shall be granted without notice to the opposite party unless it shall clearly appear from specific facts shown by affidavit or by the verified bill that immediate and irreparable injury, loss, or damage will result to the applicant before notice can be served and a hearing had thereon.
    -- Clayton Act §15

That no restraining order or injunction shall be granted by any court of the United States, or a judge or the judges thereof, in any case between an employer and employees, or between employers and employees, or between employees, or between persons employed and persons seeking employment, involving, or growing out of, a dispute concerning terms or conditions of employment, unless necessary to prevent irreparable injury to property, or to a property right, of the party making the application, for which injury there is no adequate remedy at law, and such property or property right must be described with particularity in the application, which must be in writing and sworn to by the applicant or by his agent or attorney.
    -- Clayton Act §20 (labor injunctions)
```

**Community 2, the green nodes - administrative enforcement and contempt:**

```text
The person so complained of shall have the right to appear at the place and time so fixed and show cause why an order should not be entered by the commission or board requiring such person to cease and desist from the violation of the law so charged in said complaint.
    -- Clayton Act §11 (the Interstate Commerce Commission and Federal Trade Commission machinery, which is five units on its own)

That any person who shall willfully disobey any lawful writ, process, order, rule, decree, or command of any district court of the United States or any court of the District of Columbia by doing any act or thing therein, or thereby forbidden to be done by him, if the act or thing so done by him be of such character as to constitute also a criminal offense under any statute of the United States, or under the laws of any State in which the act was committed, shall be proceeded against for his said contempt as hereinafter provided.
    -- Clayton Act §21 (contempt of court)
```

Notice that the tiler does not follow the section numbering. It pulled Sherman §1-3 into a single unit because they say almost the same thing (restraint of trade), and it scattered the Clayton sections by topic rather than by their order in the act. That is the whole point: cut along meaning, not along structure.

## The second step: carrying a change through the law

Now the part this library is really for. Say a lawmaker drafts a modern amendment about digital platforms and algorithmic pricing:

```text
It shall be unlawful for a dominant online platform, in the course of commerce, to use pricing algorithms to coordinate prices with competitors or to fix the prices of goods or services offered through the platform, where the effect may be to substantially lessen competition or tend to create a monopoly. A platform shall not discriminate in price between different business users of the same digital service where the effect may be to substantially lessen competition. No platform shall acquire, directly or indirectly, the whole or any part of the share capital or the user data of a competing platform where the effect of such acquisition may be to substantially lessen competition or tend to create a monopoly. Any person injured in business or property by reason of such conduct may sue and shall recover threefold the damages sustained, together with the cost of suit and a reasonable attorney's fee.
```

One paragraph, but it clearly touches several different provisions. Feeding it to the second step is, again, one call:

```python
from text_change_detector import detect_changes, Change

result = detect_changes(tiling, [Change(name="digital-platform-antitrust", text=amendment)])
```

The library ranks the units the change resembles plus their graph neighbours, has a local LLM rate each one, verifies the strong hits with a skeptical second pass, and drafts a merged text for the survivors. Out of 8 candidates it reviewed, it confirmed and produced edits for **five units at once, spread across the law**:

- Sherman §1 - restraint of trade and price coordination
- Clayton §2 - price discrimination between buyers
- Clayton §7 - acquiring a competitor's share capital
- Clayton §14 - corporate liability and enforcement
- Clayton §15 - the litigation procedure that would carry it

That is exactly the third pain point from the top of this post: a single change request that relates to a bunch of fragments dispersed across the document. The library found them, threw out the two hits that did not survive the skeptical pass, and drafted the merged wording for the rest. How the tiling and the ranking actually work under the hood is what the next posts are for :)

## The same tiler on Polish, and a bonus: it quarantines the junk

The tiler does not care what language the document is in, so let me point it at a Polish statute. I took a 170-sentence chunk of the Polish Code of Civil Procedure (Kodeks postępowania cywilnego) that deliberately straddles two very different kinds of text: 65 sentences of front matter ("obwieszczenie", an announcement that quotes the effective-date and transitional clauses of dozens of amending laws), followed by 105 sentences of the actual code articles (the scope of the procedure, the parties' duties, which court hears what).

`tile` split it into 42 units and 3 communities. The interesting part is not just that it works on Polish, but where the front-matter boilerplate went. It did not smear itself across the substantive articles. Almost all of it landed in a single community:

| community | units | front matter | code body |
|---|---|---|---|
| 0 | 12 | 12 | 0 |
| 1 | 17 | 3 | 14 |
| 2 | 13 | 1 | 12 |

Community 0 (the blue nodes in the picture below) is 100% the announcement's boilerplate, the effective-date and transitional clauses:

```text
Ustawa wchodzi w życie po upływie 14 dni od dnia ogłoszenia, z wyjątkiem art. 5, który wchodzi w życie z dniem następującym po dniu ogłoszenia.

Do postępowań cywilnych wszczętych i niezakończonych przed dniem wejścia w życie niniejszej ustawy stosuje się przepisy ustawy zmienianej w art. 2, w brzmieniu dotychczasowym, z wyjątkiem art. 61 § 3 ustawy zmienianej w art. 2, który stosuje się w brzmieniu nadanym niniejszą ustawą.

23) art. 18 i art. 24 ustawy z dnia 26 stycznia 2023 r. o zmianie ustaw w celu likwidowania zbędnych barier administracyjnych i prawnych (Dz. U. poz. 803), które stanowią:

Do spraw o zobowiązanie osoby stosującej przemoc w rodzinie do opuszczenia wspólnie zajmowanego mieszkania i jego bezpośredniego otoczenia lub o zakazanie zbliżania się do mieszkania i jego bezpośredniego otoczenia wszczętych i niezakończonych przed dniem wejścia w życie niniejszej ustawy stosuje się przepisy dotychczasowe.

Ustawa wchodzi w życie po upływie 3 miesięcy od dnia ogłoszenia, z wyjątkiem:
```

while communities 1 and 2 (the orange and green nodes) are the real code articles, and they cohere by subject: the scope of the procedure, the parties' duties, and which court is competent:

```text
Kodeks postępowania cywilnego normuje postępowanie sądowe w sprawach ze stosunków z zakresu prawa cywilnego, rodzinnego i opiekuńczego oraz prawa pracy, jak również w sprawach z zakresu ubezpieczeń społecznych oraz w innych sprawach, do których przepisy tego Kodeksu stosuje się z mocy ustaw szczególnych (sprawy cywilne).

Do rozpoznawania spraw cywilnych powołane są sądy powszechne, o ile sprawy te nie należą do właściwości sądów szczególnych, oraz Sąd Najwyższy.

Prokurator może żądać wszczęcia postępowania w każdej sprawie, jak również wziąć udział w każdym toczącym się już postępowaniu, jeżeli według jego oceny wymaga tego ochrona praworządności, praw obywateli lub interesu społecznego.

Strony i uczestnicy postępowania obowiązani są dokonywać czynności procesowych zgodnie z dobrymi obyczajami, dawać wyjaśnienia co do okoliczności sprawy zgodnie z prawdą i bez zatajania czegokolwiek oraz przedstawiać dowody.

Sąd powinien przeciwdziałać przewlekaniu postępowania i dążyć do tego, aby rozstrzygnięcie nastąpiło na pierwszym posiedzeniu, jeżeli jest to możliwe bez szkody dla wyjaśnienia sprawy.

Sądy rejonowe rozpoznają wszystkie sprawy z wyjątkiem spraw, dla których zastrzeżona jest właściwość sądów okręgowych.

Strony i uczestnicy postępowania mają prawo przeglądać akta sprawy i otrzymywać odpisy, kopie lub wyciągi z tych akt.
```

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/kpc-tiling-communities.png" /><br />

Three transitional clauses did leak into the code communities, so it is not surgical, but the bulk of the boilerplate is quarantined in its own bucket, away from the substantive text. That matters for the change-impact job from earlier: you do not want a new requirement to be matched against effective-date boilerplate just because it happens to live in the same PDF.

## A sanity check: the same topic, split in two

Here is the property that makes the community step worth having, and it is easy to check. I built a tiny document out of two very different topics, but I placed one of them in two spots: a block about brewing coffee, then a block about database indexes, then more about brewing coffee. Topic A at the start and the end, topic B sandwiched in the middle.

If the grouping were driven by position, the two coffee blocks would never meet, since they are nowhere near each other in the text. But the grouping is driven by meaning, so `tile` should pull them back together into one community and leave the database block on its own.

That is exactly what happens. The 27 sentences became 6 units and 2 communities, both perfectly clean:

```text
community "coffee"     -> original sentence positions 0-8 and 19-26   (both blocks, the start and the end)
community "databases"  -> original sentence positions 9-18            (the block in the middle)
```

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/two-topics-sanity.png" /><br />

The coffee sentences from the very start and the very end land in the same community (brown), with the database block (blue) kept separate, even though a wall of unrelated text stands between the two coffee halves. This is the whole reason units are clustered on a graph instead of just being cut into a linear sequence: it is what lets a single change request find every fragment it touches, wherever they sit in the document. The library ships this as an actual test, on a four-chapter document where two far-apart chapters about money must end up in the same community.

## Limitations

A few honest caveats:

First, this is not an exact same-topic detector. It groups by meaning, but it does so through two lossy steps: a segmenter that decides where units start and end, and a graph-clustering step that is a greedy heuristic. So you will see the odd unit land in a neighbouring community, and, as in the Polish run above, a few stray sentences leak out of the bucket they "should" be in. It gets the big structure right; it does not promise a perfect partition.

Second, and more fundamentally, the whole thing leans on the document being a real document. The segmentation step is a TextTiling descendant: it finds boundaries by comparing adjacent windows of text, which only means something if neighbouring sentences are actually related. Shuffle the sentences and that signal is gone. When I scrambled the order of the Polish chunk as an experiment, the front-matter-versus-code separation fell apart, not because the clustering failed but because the units it was handed were now random mixes of both. My hunch is that recovering the original topic structure from a fully shuffled document would be a hard problem even for a frontier model, since the local coherence it would otherwise lean on has been destroyed on purpose. That is a hypothesis, not something I have measured.

So the sweet spot is the opposite of a shuffle: real, non-random text where related things tend to sit near each other and different domains read differently. On that kind of input the library groups meanings quite well, which is exactly what the coffee-and-databases check above, and the antitrust and KPC runs before it, were meant to show.

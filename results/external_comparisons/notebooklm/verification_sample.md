# NotebookLM five-claim page check

This diagnostic check uses run 05 because its copied response preserved printed page references in plain text. The selection takes one page-attributed claim from each of the first five named works whose journal pagination could be mapped directly to the extracted corpus pages. This deterministic rule was used because the copied response did not preserve links between its page-looking references and NotebookLM's source interface.

| Claim | Copied reference | Corpus page checked | Result |
| --- | --- | --- | --- |
| High settler mortality encouraged extractive colonial institutions. | Acemoglu, Johnson, and Robinson, p. 1370 | `AcemogluEtAl_2001_page_2.txt` | Supported. The page describes extractive states, settlement feasibility, and persistence. |
| The Glorious Revolution constrained the Crown through a new institutional balance. | North and Weingast, p. 803 | `North&Weingast_1989_page_1.txt` | Unsupported at the cited page. The retained page is the article's cover and bibliographic page. |
| Factor endowments shaped inequality and institutional development in the New World. | Sokoloff and Engerman, p. 220 | `Sokoloff&Engerman_2000_page_4.txt` | Supported. The page connects factor endowments, inequality, and persistent institutional paths. |
| Ogilvie's conflict view treats institutions as outcomes of distributional struggle. | Ogilvie, p. 661 | `Ogilvie_2007_page_13.txt` | Unsupported at the cited page. The page discusses cultural-efficiency arguments and self-enforcing institutions. |
| Ideas and rhetoric helped produce Northwestern Europe's economic transformation. | McCloskey, p. 65 | `McCloskey_2015_page_9.txt` | Supported. The page assigns causal force to ideas, rhetoric, and their conjunction with historical circumstances. |

Three of the five checked claims were supported by the cited printed page. The two failures illustrate why a page-looking reference is insufficient without a preserved document-page-passage binding and a release check.

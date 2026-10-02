# Amazon review moderation: SME policy (summary only)

The full SME policy F is private (`var/policy/amazon_reviews_F.txt`, 1,067 words,
sha256 `9fa66176206b4fead64a324e8c3ab79b6d78dc7659804c78c9cc0e495148fc2d`). It was written in our own
words for this study, informed by two public Amazon help pages fetched 2026-10-02 and kept only under
`var/policy/raw/` (not redistributed):

| Page | URL | sha256 of the fetched HTML |
| --- | --- | --- |
| Community Guidelines | https://www.amazon.com/gp/help/customer/display.html?nodeId=GLHXEX85MENUE4XF | `b535a0e7bbd34cb3551919603a4c5670fcb8c57b040d788684b43b83c85ec825` |
| Customer reviews (anti-manipulation) | https://www.amazon.com/gp/help/customer/display.html?nodeId=G3UA5WC5S5UUKB5G | `17b0f5d5b698baab1f4a9d83db82f7e58f2ea17d34423a8d8bda3e9ed809b9f0` |

The classifier never sees F; it starts from the one-liner in `S.txt`.

## Shape of F

- Five labels with a fixed precedence: abusive, then promotional, then seller_shipping, then
  price_availability, then approve. The highest applicable one wins.
- Twelve numbered rules, R1-R12; every SME decision names exactly one.
  - R1-R2 abusive: swearing (mild swear words included, harsh-but-clean words excluded); attacks on people.
  - R3-R5 promotional: contact details or links; any disclosure of a free or cheaper item given for the
    review; pushing another brand, store or the writer's own business.
  - R6-R7 seller_shipping: the review is mainly about the transaction (seller, service, refunds, wrong
    item) or about delivery (speed, cost, packaging, damage in transit).
  - R8-R9 price_availability: store-, time- or deal-specific prices; stock at a particular shop.
  - R10-R12 approve: value-for-money judgements, product-focused reviews (including angry ones and
    products that failed in use), and product reviews with only a passing transaction remark.
- A "main point" test separates seller_shipping from approve; abusive and promotional apply on any
  occurrence; a store-specific price claim applies even when brief.
- The SME may answer `ambiguous` (rarely) and must give its reason in its own words.

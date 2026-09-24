from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

tesla_text = """

MGI represents an enterprise and brand with 51 years of national and global experience. It currently operates more than 57 industrial units, houses over 65,000 employees, 6,650 distributors, and 20,000 suppliers under its umbrella.

The history of one of Bangladesh’s largest leading conglomerates, Meghna Group of Industries (MGI) can be traced all the way back to 1976 when its predecessor operated under the name of Kamal Trading Company. The conglomerate itself has humble origins and began its life as Meghna Vegetable Oil Industries Ltd. in 1989 on a small patch of land in Meghnaghat, Narayanganj.

The secret to the success and vast expansion of MGI has been diversification. The group has entered a broad array of different markets and industries including Fast Moving Consumer Goods (FMCG), building materials, pulp and paper, LPG, feeds, fiber, power plants, shipping, seeds crushing, chemicals, ship building, dockyard, securities, insurance, media and aviation. The product range of MGI today is truly impressive and the conglomerate markets most of its products under the recognisable brand names of "Fresh", "No.1", "Actifit", "Pure" and "Meghnacem Deluxe". The result of this level of reach and diversification has been that one in every two households in Bangladesh uses MGI products. Internationally MGI has a substantial presence in the Middle East, Southeast Asia, Europe, South Africa, and North and South America.

As a result of this relentless process of expansion MGI has become a powerful player within Bangladesh and has become the largest investor in relation to industrial development in Bangladesh over the last few years. MGI became the first company in Bangladesh to establish a private economic zone known as the "Meghna Economic Zone", which has since been followed by the creation of three further economic zones, which are named “Meghna Industrial Economic Zone”, “Cumilla Economic Zone” and "Titas Economic Zone" respectively. The conglomerate has expanded even further since this point, with an unprecedented investment of $451 million in 2020 that has erected nine new industrial units within its multiple economic zones.

 Throughout this process the unwavering commitment of its visionary leader, Mostafa Kamal, has been pivotal for both the conglomerate and the Bangladeshi Economy. Renowned for his entrepreneurial expertise and patriotism, Mostafa Kamal has played a key role in the development of industry, healthcare, education, sports and social welfare in Bangladesh. The integrity and dedication towards the group that he has played a vital part in the overall success of MGI.
 
 
"""


# splitter1 = CharacterTextSplitter(
#     separator=" ",  # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
#     chunk_size=100,
#     chunk_overlap=0
# )

# chunks1 = splitter1.split_text(tesla_text)
# for i, chunk in enumerate(chunks1, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f'"{chunk}"')
#     print()



# Example 2: RecursiveCharacterTextSplitter fixes this
print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # Multiple separators
    chunk_size=600,
    chunk_overlap=0
)

chunks2 = recursive_splitter.split_text(tesla_text)
print(f"Same problem text, but with RecursiveCharacterTextSplitter:")
for i, chunk in enumerate(chunks2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400, chunk_overlap = 0
)

## it used text structure like paragraph, lines, words and character.
## recursively ceate chunk as per Paragraph then lines then words, then character untill chunk size is acheived

text = '''Nature is the encompassing physical world and life, from subatomic particles to cosmic phenomena, providing essential resources like fresh air, water, and food, while also offering beauty, peace, and inspiration through its diverse landscapes, seasons, and creatures, though it faces threats from human activities like pollution and deforestation, making its preservation a critical duty for sustaining all life.

        Nature is the fundamental force that sustains life, encompassing all living and non-living things, from towering mountains and vast oceans to the smallest microbes and plants. It provides humans with necessities like oxygen, clean water, and food, while regulating vital processes such as decomposition and flood control. The beauty of nature, seen in sunrises, singing birds, and tranquil forests, offers rest and rejuvenation from daily stresses, highlighting the interconnectedness of ecosystems.'''
docs = splitter.split_text(text)
print(len(docs))
print(docs)
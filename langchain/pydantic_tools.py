from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from core.llm_chain import PydanticToolChain


class TaggingJob:

    class Tagging(BaseModel):
        """Tag the piece of text with particular info."""

        sentiment: str = Field(
            description="The sentiment of the text, should be `pos`, `neg`, or `neutral`."
        )
        language: str = Field(
            description="The language of the text, should be ISO 639-1 code."
        )

    @staticmethod
    def tagging() -> None:
        chain = PydanticToolChain(TaggingJob.Tagging)
        chain.invoke("non mi piace questo cibo!")


class PersonJob:

    class Person(BaseModel):
        """Extract the information of the person."""
        name: str = Field(
            description="The name of the person."
        )
        age: Optional[int] = Field(
            description="The age of the person."
        )
        
    class PersonInfo(BaseModel):
        """Extract the information of the person."""
        people: List["PersonJob.Person"] = Field(
            description="list of info about people"
        )

    @staticmethod
    def extract_person() -> None:
        chain = PydanticToolChain(PersonJob.PersonInfo)
        chain.invoke("Joe is 30, and Jane is 25. Kite is their mother.")


class ArticleJob:

    content: str = (Path(__file__).resolve().parent / "resources" / "lilianweng_posts_2023-06-23-agent_content.txt").read_text(encoding="utf-8")

    class ArticleSummary(BaseModel):
        """Overview of a section of article."""
        summary: str = Field(
            description="Provide a concise summary of the article."
        )
        language: str = Field(
            description="Provide the language of the article."
        )
        keywords: List[str] = Field(
            description="Provide keywords related to the article."
        )

    class PaperInfo(BaseModel):
        """Extract the information of the paper."""
        title: str = Field(
            description="Title of the paper."
        )
        author: Optional[str] = Field(
            description="Author of the paper."
        )

    class PaperInfos(BaseModel):
        """Extract the information of the papers."""
        papers: List["ArticleJob.PaperInfo"] = Field(
            description="List info about papers"
        )

    @staticmethod
    def summary() -> None:
        chain = PydanticToolChain(ArticleJob.ArticleSummary)
        chain.invoke(ArticleJob.content)

    @staticmethod
    def extract_paper() -> None:
        system_prompt = "A article will be passed to you. Extract from it all papers that are metioned by this article. Do not extract the name of the article itself. If no papers are mentioned, return an empty list. Do not make up or guess ANY extra information. Only extract what exactly is in the text. "
        chain = PydanticToolChain(ArticleJob.PaperInfos, system_prompt)
        chain.invoke(ArticleJob.content, split=True)


def main() -> None:
    TaggingJob.tagging()
    PersonJob.extract_person()
    ArticleJob.summary()
    ArticleJob.extract_paper()


if __name__ == "__main__":
    main()

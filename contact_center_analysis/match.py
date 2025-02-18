from typing import List, Dict, Any, Tuple, Optional
from .base import BaseAnalyzer

class Matcher(BaseAnalyzer):
    """Match and compare text content using semantic similarity."""
    
    async def find_matches(self,
                         query: str,
                         candidates: List[str],
                         top_k: int = 1,
                         threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find semantic matches for a query string from a list of candidates.

        Args:
            query: The string to match against
            candidates: List of potential matches
            top_k: Number of top matches to return
            threshold: Minimum similarity score (0-1) to consider a match

        Returns:
            List of dicts containing match info (text, score, rank)
        """
        prompt = f"""Compare the semantic similarity between the query and each candidate.
        
Query: "{query}"

Candidates:
{self._format_candidates(candidates)}

Return JSON with matches above {threshold} similarity, including:
- text: The matching candidate text
- score: Similarity score (0-1)
- rank: Ranking position
- rationale: Brief explanation of the semantic relationship

Sort by similarity score descending. Return at most {top_k} matches."""

        response = await self._generate_content(
            prompt,
            expected_format={
                "matches": [{
                    "text": str,
                    "score": float,
                    "rank": int,
                    "rationale": str
                }]
            }
        )
        
        return response["matches"]

    async def group_by_similarity(self,
                                items: List[str],
                                min_similarity: float = 0.8) -> List[List[str]]:
        """Group items by their semantic similarity.

        Args:
            items: List of strings to group
            min_similarity: Minimum similarity score to group items together

        Returns:
            List of groups (each group is a list of similar items)
        """
        prompt = f"""Group these items based on semantic similarity.
        
Items:
{self._format_candidates(items)}

Create groups where items have at least {min_similarity} similarity score.
Return JSON with:
- groups: List of groups
- rationale: Explanation for each group

Each group should have:
- items: List of items in the group
- theme: Short description of the common theme"""

        response = await self._generate_content(
            prompt,
            expected_format={
                "groups": [{
                    "items": [str],
                    "theme": str,
                    "rationale": str
                }]
            }
        )
        
        return response["groups"]

    def _format_candidates(self, items: List[str]) -> str:
        """Format list of items for prompt."""
        return "\n".join(f"- {item}" for item in items) 
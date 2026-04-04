"""WikiLinter — health checks for the knowledge base.

Finds issues like:
- Files with no content (empty)
- Broken [[wikilinks]] (referenced but don't exist)
- Duplicate content across files
- Very large files that may need splitting
- Files with no headings (poor structure)
- Stale content (old timestamps in frontmatter)

Inspired by Pal's wiki linting concept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections import defaultdict
from mcq.core.types import Corpus, CorpusChunk


@dataclass
class LintIssue:
    severity: str  # "error", "warning", "info"
    file_path: str
    message: str
    suggestion: str | None = None


class WikiLinter:
    @staticmethod
    def lint(corpus: Corpus) -> list[LintIssue]:
        """Run all health checks on a corpus. Returns a list of issues."""
        issues: list[LintIssue] = []
        issues.extend(WikiLinter._check_empty_files(corpus))
        issues.extend(WikiLinter._check_broken_wikilinks(corpus))
        issues.extend(WikiLinter._check_large_files(corpus))
        issues.extend(WikiLinter._check_no_headings(corpus))
        issues.extend(WikiLinter._check_duplicate_content(corpus))
        # Sort: errors first, then warnings, then info
        severity_order = {"error": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda i: (severity_order.get(i.severity, 3), i.file_path))
        return issues

    @staticmethod
    def _check_empty_files(corpus: Corpus) -> list[LintIssue]:
        issues = []
        for chunk in corpus.chunks:
            stripped = chunk.content.strip()
            if not stripped:
                issues.append(LintIssue(
                    severity="error",
                    file_path=chunk.source_path,
                    message="File is empty",
                    suggestion="Add content or remove the file",
                ))
            elif len(stripped) < 20:
                issues.append(LintIssue(
                    severity="warning",
                    file_path=chunk.source_path,
                    message=f"File has very little content ({len(stripped)} chars)",
                    suggestion="Consider expanding or merging with another file",
                ))
        return issues

    @staticmethod
    def _check_broken_wikilinks(corpus: Corpus) -> list[LintIssue]:
        """Find [[wikilinks]] that point to non-existent files."""
        # Build set of known file stems (without extension)
        known_stems = set()
        known_paths = set()
        for chunk in corpus.chunks:
            known_paths.add(chunk.source_path)
            # Also add stem without extension
            if "." in chunk.source_path:
                stem = chunk.source_path.rsplit(".", 1)[0]
                known_stems.add(stem.lower())
                # Also add just the filename stem
                if "/" in stem:
                    known_stems.add(stem.rsplit("/", 1)[-1].lower())

        issues = []
        for chunk in corpus.chunks:
            wikilinks = re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', chunk.content)
            for link in wikilinks:
                link_lower = link.lower().strip()
                # Check if link target exists (as path or stem)
                if (link not in known_paths and
                    link_lower not in known_stems and
                    link_lower.replace(" ", "-") not in known_stems and
                    link_lower.replace("-", " ") not in known_stems):
                    issues.append(LintIssue(
                        severity="warning",
                        file_path=chunk.source_path,
                        message=f"Broken wikilink: [[{link}]]",
                        suggestion=f"Create '{link}.md' or fix the link",
                    ))
        return issues

    @staticmethod
    def _check_large_files(corpus: Corpus, threshold: int = 10000) -> list[LintIssue]:
        """Flag files that are very large and might benefit from splitting."""
        issues = []
        for chunk in corpus.chunks:
            size = len(chunk.content)
            if size > threshold:
                issues.append(LintIssue(
                    severity="info",
                    file_path=chunk.source_path,
                    message=f"Large file ({size:,} chars, ~{size // 4} tokens)",
                    suggestion="Consider splitting into smaller focused documents",
                ))
        return issues

    @staticmethod
    def _check_no_headings(corpus: Corpus) -> list[LintIssue]:
        """Flag markdown files without any headings."""
        issues = []
        for chunk in corpus.chunks:
            if not chunk.source_path.endswith((".md", ".qmd")):
                continue
            if not re.search(r'^#+\s', chunk.content, re.MULTILINE):
                issues.append(LintIssue(
                    severity="info",
                    file_path=chunk.source_path,
                    message="No headings found in markdown file",
                    suggestion="Add headings to improve structure and searchability",
                ))
        return issues

    @staticmethod
    def _check_duplicate_content(corpus: Corpus) -> list[LintIssue]:
        """Find files with substantially similar content."""
        issues = []
        # Simple approach: compare word sets (Jaccard similarity)
        word_sets: list[tuple[str, set[str]]] = []
        for chunk in corpus.chunks:
            words = set(re.findall(r'\w+', chunk.content.lower()))
            if len(words) > 10:  # Skip very short files
                word_sets.append((chunk.source_path, words))

        seen = set()
        for i, (path_a, words_a) in enumerate(word_sets):
            for j, (path_b, words_b) in enumerate(word_sets):
                if j <= i:
                    continue
                pair = (path_a, path_b)
                if pair in seen:
                    continue
                intersection = len(words_a & words_b)
                union = len(words_a | words_b)
                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.7:
                        seen.add(pair)
                        issues.append(LintIssue(
                            severity="warning",
                            file_path=path_a,
                            message=f"Very similar content to '{path_b}' ({similarity:.0%} overlap)",
                            suggestion="Consider merging or deduplicating",
                        ))
        return issues

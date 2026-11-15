from sonata.build_library import best_match, normalize, same_artist


def result(id, title, artist):
    return {"id": id, "title": title, "artist": {"name": artist}, "readable": True}


def test_normalize_drops_version_tags_but_keeps_remix():
    assert normalize("Headlines (Explicit Version)") == "headlines"
    assert normalize("Tití Me Preguntó") == "titi me pregunto"
    assert normalize("Song - 2011 Remaster") == "song"
    assert normalize("Song (feat. Someone)") == "song"
    assert normalize("ALELUYA - REMIX") == normalize("ALELUYA (REMIX)") == "aleluya remix"


def test_same_artist_tolerates_punctuation_only():
    assert same_artist("Fred again..", "Fred again")
    assert same_artist("Arcángel", "Arcangel")
    assert not same_artist("Drake", "Drake Milligan")
    assert not same_artist("Drake", "Rockabye Baby!")


def test_best_match_prefers_the_real_artist_over_covers():
    results = [result(1, "Headlines (Made Famous By Drake)", "Young Money Millionaires"),
               result(2, "Headlines", "Rockabye Baby!"),
               result(3, "Headlines (Explicit Version)", "Drake")]
    assert best_match(results, "Drake", "Headlines")["id"] == 3


def test_best_match_returns_none_when_only_covers_exist():
    results = [result(1, "Fireworks (Originally By Drake & Alicia Keys Karaoke)", "HOT 100"),
               result(2, "FIREWORKS", "Veen")]
    assert best_match(results, "Drake", "Fireworks") is None


def test_best_match_rejects_a_different_song_by_the_same_artist():
    assert best_match([result(1, "F*****g Fans", "Drake")], "Drake", "Fancy") is None


def test_best_match_accepts_a_featured_artist_listed_first_on_deezer():
    results = [result(1, "Collab Song", "Guest Artist")]
    assert best_match(results, "Main Artist", "Collab Song") is None
    assert best_match(results, "Main Artist", "Collab Song", also_by=["Guest Artist"])["id"] == 1

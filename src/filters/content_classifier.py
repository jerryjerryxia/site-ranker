"""
Content type classification for domains.

Classifies domains into content categories:
- video_streaming: Movie/TV piracy sites (our target)
- adult: Adult/pornographic content
- file_hosting: Cyberlockers, file sharing (rapidgator, mega, etc.)
- social: Social media, user-generated content platforms
- unknown: Cannot determine

Uses multiple strategies:
1. Domain blocklists (UT1, StevenBlack, Shallalist)
2. Domain pattern matching (regex)
3. Content keyword analysis (from scraped text)
4. Site characteristic analysis (traffic patterns, metadata)
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Set


class ContentType(Enum):
    """Content type categories."""

    VIDEO_STREAMING = "video_streaming"  # Movie/TV piracy (TARGET)
    ADULT = "adult"  # Adult content
    FILE_HOSTING = "file_hosting"  # Cyberlockers, file sharing
    EBOOK = "ebook"  # Ebook/book piracy (Z-Library, etc.)
    SOCIAL = "social"  # Social media, UGC platforms
    LEGITIMATE = "legitimate"  # Known legitimate sites (Patreon, etc.)
    UNKNOWN = "unknown"  # Cannot determine


@dataclass
class ContentClassification:
    """Result of content classification."""

    content_type: ContentType
    confidence: float  # 0.0-1.0
    reasons: list[str]  # Evidence for classification
    is_target: bool  # True if video_streaming OR file_hosting (both are targets)


class ContentClassifier:
    """
    Multi-strategy content type classifier.

    Combines blocklists, pattern matching, and keyword detection
    to classify domains by content type.
    """

    def __init__(self, blocklist_dir: Optional[Path] = None):
        """
        Initialize content classifier.

        Args:
            blocklist_dir: Directory containing blocklist files
                          (defaults to data/blocklists/)
        """
        if blocklist_dir is None:
            blocklist_dir = Path(__file__).parent.parent.parent / "data" / "blocklists"

        self.blocklist_dir = blocklist_dir

        # Load blocklists
        self.adult_domains: Set[str] = set()
        self.social_domains: Set[str] = set()
        self.ebook_domains: Set[str] = set()

        # Hardcoded blocklists from GTR analysis
        self._init_hardcoded_blocklists()

        # Social media and legitimate site lists
        self._init_social_and_legit_lists()

        # Will be loaded lazily on first use
        self._blocklists_loaded = False

        # Domain patterns (regex)
        self._compile_patterns()

        # Keywords for content detection
        self._define_keywords()

    def _init_hardcoded_blocklists(self):
        """Initialize hardcoded blocklists from GTR data analysis."""

        # Known adult sites from GTR top 100
        self.hardcoded_adult = {
            'nudogram.com', 'picazor.com', 'xxbrits.com', 'hotzxgirl.com',
            'cums.net', 'nudostar.tv', 'fapello.com', 'thefappening.plus',
            'erothots.co', 'sorrymother.video', 'faponic.com', 'masterfap.net',
            'coomer.su', 'thefap.net', 'camhub.cc', 'thefappeningblog.com',
            'myxxgirl.com', 'leakedmodels.com', 'fapeza.com', 'pornx.to',
            'thothd.com', 'leakedzone.com', 'twpornstars.com', 'ibradome.com',
            'camwhores.tv', 'fapodrop.com', 'nsfw.xxx', 'sexiezpix.com',
            'leaknudes.com', 'fappeningbook.com', 'tnaflix.com', 'erothots1.com',
            'camshowrecordings.com', 'x-fetish.tube', 'x-tg.tube',
            'xvideosgostosa.com', 'camshowrecord.net', 'erome.tv',
        }

        # Known Z-Library / ebook sites (many use random domains)
        # These are confirmed ebook aggregators, not video streaming
        self.hardcoded_ebook = {
            'z-library.sk', 'z-lib.blog', 'z-lib.sx', 'z-library.ec',
            'z-lib.digital', 'openzlib.link', 'z-lib.today', 'z-lib.gd',
            'z-lib.life', 'z-lib.fo', 'z-lib.fm', 'z-lib.gl', 'z-lib.help',
            'urdu-books.sk', 'annas-archive.org', 'grababook.shop',
            'annas-archive.se', 'annas-archive.li', 'zlib.by', 'z-lib.world',
            'z-lib.shop', 'z-lib.rest', 'booklibrary.cc', 'z-library.website',
            'z-library.hn', 'z-library.la', 'zbook.in', 'zlibrary.sk',
            'libgen.rs', 'libgen.is', 'libgen.st', 'b-ok.cc', 'booksc.org',
            '1lib.sk', 'zlih.ru', 'zliw.ru', 'fdlib.com', 'b-i-l-z.de',
            'zzlib.ru', 'zlibrarya.online', 'zlibraryb.online', 'zlibraryo.online',
            'zlib.fi', 'z-lib.by', 'zlibdie.online', 'zlibraryr.ru',

            # Z-lib front domains (confirmed from GTR analysis)
            'articles.sk', '101ml.fi', '101ml.life', '101o.ru', '101su.ru',
            '101y.ru', '101l.online', '101i.online', '101j.online', '101mon.online',
            '101o.online', '101p.online', 'quasieconomist.online', 'fudanedu.online',
            'zlib520.online', 'z1412.online', 'z8341.online', 'liaoyuan.online',
            'niulangshan.online', 'veritasforall.online', 'veritasforall.store',
            'zlib4.online', 'zlibn.online', 'annas-archive.online',
            'linuxjournal.press', 'quacio.us', 'pyxlovewyf.com', 'l1is.gg',
            'ws3917.site', 'pdffree.store', 'writesomething.sbs', 'gopi.monster',
            'leftist.ru', 'actrue.fun', 'sun4mile.asia', 'sgnilo.wiki',
            'k3v2yu.top', 'll1.my', 'rhonin.top', 'polyu.news', 'xuyadong.sbs',
            'educateyourself.ly', 'pkuedu.ru', 'zeegwet.shop', 'tianxingjian.wiki',
            'doceru.online', 'borless.com', 'publicintellect.com', 'librerialibre.eu',
            'ch8.fm', 'filifili.cf', 'anin.click', 'youngmoneygpt.com', 'lhbooks.top',
            'dormessa.com', 'pustakalya.vip', 'blyat.su', 'e-book.cash', 'shoujo.top',
            'rnm.ink', 'sac.org.ar', 'velvetvixen.shop', '491001.online',
            'onlyforscholarship.online', 'ugda.online', 'free2read.cc', 'allfree2.me',

            # Additional Z-lib mirrors discovered
            'freeepubs.com', 'leninist.online', 'chishui.online', '191954.online',
            'marwanshop.online', '101e.online', 'zz101.online', 'bluesky-travel.fr',
            'freeda.com.br', 'zliv.online',
        }

    def _init_social_and_legit_lists(self):
        """Initialize social media and legitimate site lists."""

        # Major social media platforms
        self.hardcoded_social = {
            # Core social networks
            'facebook.com', 'instagram.com', 'twitter.com', 'x.com',
            'tiktok.com', 'linkedin.com', 'pinterest.com', 'snapchat.com',
            'threads.net', 'mastodon.social', 'bsky.app', 'bluesky.social',

            # Video platforms
            'youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com',
            'twitch.tv', 'kick.com',

            # Messaging platforms
            'discord.com', 'discordapp.com', 'telegram.org', 't.me',
            'whatsapp.com', 'signal.org', 'slack.com',

            # Forum/community platforms
            'reddit.com', 'quora.com', 'tumblr.com', 'medium.com',
            'substack.com', 'livejournal.com',

            # Chinese social platforms
            'weibo.com', 'wechat.com', 'qq.com', 'douyin.com',
            'bilibili.com', 'bilibili.tv', 'xiaohongshu.com', 'zhihu.com',

            # Russian social platforms
            'vk.com', 'vkvideo.ru', 'ok.ru', 'mail.ru',

            # Japanese platforms
            'line.me', 'ameba.jp', 'nicovideo.jp',

            # Other regional social
            'kakaotalk.com', 'naver.com', 'zalo.me',
        }

        # Known legitimate sites (platforms that receive takedowns but aren't piracy)
        self.hardcoded_legitimate = {
            # Creator platforms
            'patreon.com', 'ko-fi.com', 'buymeacoffee.com', 'gumroad.com',
            'fanbox.cc', 'subscribestar.com', 'boosty.to',

            # Major tech/cloud platforms
            'google.com', 'drive.google.com', 'docs.google.com',
            'microsoft.com', 'onedrive.com', 'office.com', 'live.com',
            'apple.com', 'icloud.com',
            'amazon.com', 'aws.amazon.com',
            'dropbox.com', 'box.com',

            # Development platforms
            'github.com', 'gitlab.com', 'bitbucket.org',
            'stackoverflow.com', 'sourceforge.net',

            # Document/publishing platforms
            'issuu.com', 'scribd.com', 'slideshare.net', 'academia.edu',

            # Legitimate streaming services
            'netflix.com', 'hulu.com', 'disneyplus.com', 'hbomax.com',
            'primevideo.com', 'peacocktv.com', 'paramount.com',
            'crunchyroll.com', 'funimation.com', 'vrv.co',
            'spotify.com', 'soundcloud.com', 'bandcamp.com',
            'pluto.tv', 'tubi.tv', 'roku.com', 'plex.tv',
            'france.tv', 'rtp.pt', 'bbc.com', 'bbc.co.uk',

            # News/media outlets
            'cnn.com', 'bbc.com', 'nytimes.com', 'washingtonpost.com',
            'theguardian.com', 'forbes.com', 'bloomberg.com', 'reuters.com',
            'huffpost.com', 'buzzfeed.com', 'vice.com',

            # Entertainment/gaming news
            'ign.com', 'gamespot.com', 'kotaku.com', 'polygon.com',
            'tvinsider.com', 'comicbook.com', 'fandom.com', 'imdb.com',
            'rottentomatoes.com', 'metacritic.com',

            # E-commerce
            'ebay.com', 'etsy.com', 'alibaba.com', 'aliexpress.com',
            'shopify.com', 'walmart.com', 'target.com', 'bestbuy.com',

            # Telecom/ISPs
            'claro.com.br', 'optus.com.au', 'att.com', 'verizon.com',
            't-mobile.com', 'vodafone.com', 'telefonica.com',

            # Education
            'edu', 'ac.uk', 'psu.edu', 'mit.edu', 'stanford.edu',
            'coursera.org', 'edx.org', 'udemy.com', 'khanacademy.org',

            # Government
            'gov', 'state.gov', 'gov.uk', 'europa.eu',

            # Gaming platforms
            'steam.com', 'steampowered.com', 'epicgames.com', 'gog.com',
            'playstation.com', 'xbox.com', 'nintendo.com',
            'roblox.com', 'minecraft.net', 'ea.com', 'ubisoft.com',

            # Crypto/finance
            'pump.fun', 'coinbase.com', 'binance.com', 'kraken.com',
            'paypal.com', 'stripe.com', 'venmo.com',

            # Mobile app stores
            'play.google.com', 'apps.apple.com', 'aptoide.com', 'myket.ir',
            'apkpure.com', 'uptodown.com',

            # Media software
            'kodi.tv', 'vlc.com', 'videolan.org', 'plex.tv',
            'jellyfin.org', 'emby.media',

            # Major tech companies (receive takedowns for false positives)
            'nvidia.com', 'amd.com', 'intel.com',
            'adobe.com', 'autodesk.com', 'blender.org',
            'bluestacks.com', 'noxplayer.com',
            'wetransfer.com', 'transfernow.net',
            'media-amazon.com', 'cloudfront.net', 'akamaihd.net',
            'msn.com', 'bing.com', 'outlook.com',
            'odoo.com', 'salesforce.com', 'hubspot.com',
            'twimg.com', 'fbcdn.net', 'cdninstagram.com',
            'digitaltrends.com', 'techradar.com', 'tomsguide.com', 'cnet.com',

            # News portals by region
            'elcomercio.pe', 'nld.com.vn', 'sohu.com', 'rte.ie',
            'globo.com', 'uol.com.br', 'terra.com.br',
            'thehindu.com', 'indianexpress.com', 'ndtv.com',
            'aljazeera.com', 'arabianbusiness.com',

            # Sports/esports
            'hltv.org', 'esportsearnings.com', 'liquipedia.net',

            # Legitimate music/audio
            'jamendo.com', 'freemusicarchive.org', 'audiojungle.net',

            # Legitimate game-related
            'gamer.com.tw', 'pcgamer.com', 'rockpapershotgun.com',
            'nexusmods.com', 'moddb.com', 'curseforge.com',

            # Ad/CDN networks (not piracy hosts)
            'doubleclick.net', 'googlesyndication.com', 'googleadservices.com',
            'adsrvr.org', 'criteo.com', 'taboola.com', 'outbrain.com',

            # Note-taking and paste services
            'anotepad.com', 'pastebin.com', 'hastebin.com', 'privatebin.net',

            # Old web hosting
            'tripod.com', 'angelfire.com', 'geocities.ws',
            'weebly.com', 'wix.com', 'squarespace.com', 'wordpress.com',

            # Romanian media
            'cinemagia.ro', 'film-info.ro',

            # Legitimate APK sources
            'apkpure.net', 'apkmirror.com', 'f-droid.org',

            # Review/business platforms
            'trustpilot.com', 'yelp.com', 'tripadvisor.com', 'glassdoor.com',

            # Photography/stock platforms
            '500px.com', 'unsplash.com', 'pexels.com', 'flickr.com',
            'shutterstock.com', 'gettyimages.com', 'istockphoto.com',

            # Developer/documentation sites
            'csdn.net', 'jianshu.com', 'gitee.com',

            # Language learning
            'italki.com', 'duolingo.com', 'babbel.com',

            # Software download (legitimate)
            'filehippo.com', 'softonic.com', 'download.cnet.com',

            # E-commerce (regional)
            'amazon.com.tr', 'amazon.de', 'amazon.co.uk', 'amazon.co.jp',
            'mercadolivre.com.br', 'mercadolibre.com.ar', 'fravega.com',

            # News/media (regional)
            'infobae.com', 'pikabu.ru',

            # Cloud/enterprise
            'sharepoint.com', 'teams.microsoft.com', 'notion.so', 'airtable.com',

            # Link aggregators (legitimate)
            'linklist.bio', 'linktr.ee', 'bio.link',

            # Video/streaming (legitimate)
            'rumble.com', 'odysee.com',  # User-generated, not piracy hosts
            'megogo.net', 'directvgo.com',  # Legitimate streaming services

            # Web hosting/builders
            'jimdofree.com', 'jimdo.com', 'strikingly.com', 'webnode.com',

            # Twitter CDN
            't.co',  # Twitter short links

            # Chinese cloud storage (can be abused but primarily legitimate)
            'quark.cn', 'baidu.com', 'pan.baidu.com',

            # Document viewers
            'fliphtml5.com', 'anyflip.com', 'calaméo.com',

            # Wikipedia/reference
            'wikipedia.org', 'wikimedia.org', 'wiktionary.org',
            'britannica.com', 'archive.org',

            # Search engines
            'bing.com', 'duckduckgo.com', 'yahoo.com', 'yandex.com',
            'baidu.com',

            # Tech news/reviews
            'theverge.com', 'techcrunch.com', 'wired.com', 'arstechnica.com',
            'engadget.com', 'cnet.com', 'computerbild.de', 'chip.de',
            'fernsehserien.de', 'filmstarts.de',

            # Image hosting (legit)
            'imgur.com', 'flickr.com', 'unsplash.com', 'pexels.com',
            'shutterstock.com', 'gettyimages.com',

            # News aggregators
            'news.google.com', 'news.yahoo.com', 'feedly.com',
            'flipboard.com', 'pocket.com',

            # Podcasting
            'anchor.fm', 'podbean.com', 'libsyn.com', 'transistor.fm',
            'podcasts.apple.com', 'open.spotify.com',

            # Personal blogs/websites (common TLDs that host DMCA targets)
            'wordpress.com', 'blogger.com', 'blogspot.com', 'wix.com',
            'squarespace.com', 'weebly.com', 'ghost.io',

            # Argentina sites (from top 30)
            'cronista.com', 'eldestapeweb.com',
        }

    def _compile_patterns(self):
        """Compile regex patterns for domain matching."""

        # Adult content patterns (expanded from GTR data analysis)
        self.adult_patterns = [
            # Explicit keywords (NO word boundaries - these appear in compound words)
            r'porn', r'xxx', r'sex(?:y|iez|ual)?', r'adult',
            r'nude', r'nsfw', r'erotic', r'lewd',
            r'slut', r'whore', r'fuck', r'pussy',

            # Teen/age-related adult
            r'teen.*18', r'18.*teen', r'teengirl', r'18hd',

            # Leak/fap sites (thefappening, fapello, etc.)
            r'fap', r'thot', r'leak(?:ed)?',

            # Cam/streaming adult
            r'cam(?:hub|whore|show|record)', r'tube.*(?:pussy|xxx|porn)',

            # OnlyFans leaks
            r'onlyfan', r'fansly', r'coomer',

            # Other adult patterns from GTR data
            r'nudo', r'erothot', r'hotzx', r'xxbrit',
            r'picazor', r'masterfap', r'faponic',
            r'fapeza', r'fapodrop', r'ibradome',
            r'sexiez', r'tnaflix', r'cums\.',
            r'thefap', r'myxxgirl', r'sorrymother',
            r'xvideo', r'buceta',  # buceta = Portuguese for vagina

            # Dating/escort
            r'escort', r'hookup', r'dating',

            # Major adult tube sites
            r'xnxx', r'xhamster', r'youporn', r'tube8',
            r'redtube', r'pornhub', r'spankwire',
        ]

        # Ebook/Z-Library patterns (VERY common random domain pattern)
        self.ebook_patterns = [
            # Z-lib variations
            r'\bzlib', r'\bz-lib\b', r'\bzlibrary\b', r'\b1lib\b',

            # Book-related
            r'\bbook', r'\bebook\b', r'\be-book\b', r'\bepub', r'\bpdf',
            r'\blibrary\b', r'\blibgen\b', r'\banna.*archive',

            # Z-lib common patterns (101X.online, zlXX.online, etc.)
            r'^101[a-z0-9]{1,3}\.', r'^zl[a-z]{2,4}\.', r'^\d{5,}\.online$',

            # "Free" book sites
            r'freepub', r'freeebook', r'freedownload',
        ]

        # File hosting patterns
        self.filehost_patterns = [
            r'\bupload\b', r'\bfile\b', r'\bshare\b', r'\bhost\b',
            r'\bdrive\b', r'\bcloud\b', r'\bstorage\b',
            r'\bzippyshare\b', r'\bmega\b', r'\brapid',
            r'\bnitro', r'\bturbof', r'\bup?load',
        ]

        # Video streaming patterns
        self.streaming_patterns = [
            r'\bstream\b', r'\bmovie', r'\bfilm', r'\btv\b',
            r'\bshow', r'\bwatch\b', r'\bflix\b', r'\bseries\b',
            r'\bepisode', r'\bcinema\b', r'\b\d+movies?\b',
            r'\bhdstream', r'\bfree.*watch', r'\bonline.*movie',
            # Common streaming site patterns
            r'fmovies', r'soap2day', r'putlocker', r'yesmovies',
            r'gomovies', r'primewire', r'vumoo', r'solarmovie',
            r'popcornflix', r'tubi', r'crackle',
            # Piracy-specific patterns
            r'torrent', r'download.*movie', r'hd.*movie',
        ]

        # Social media patterns
        self.social_patterns = [
            r'\bfacebook\b', r'\btwitter\b', r'\binstagram\b',
            r'\btiktok\b', r'\byoutube\b', r'\breddit\b',
            r'\bdiscord\b', r'\bsocial\b', r'\bchat\b',
        ]

        # Compile all patterns
        self.adult_regex = re.compile('|'.join(self.adult_patterns), re.IGNORECASE)
        self.ebook_regex = re.compile('|'.join(self.ebook_patterns), re.IGNORECASE)
        self.filehost_regex = re.compile('|'.join(self.filehost_patterns), re.IGNORECASE)
        self.streaming_regex = re.compile('|'.join(self.streaming_patterns), re.IGNORECASE)
        self.social_regex = re.compile('|'.join(self.social_patterns), re.IGNORECASE)

    def _define_keywords(self):
        """Define keyword sets for content analysis."""

        # Adult content keywords
        self.adult_keywords = {
            'porn', 'xxx', 'adult', 'nude', 'nsfw', 'sex', 'erotic',
            'onlyfans', 'fansly', 'patreon', 'leaked', 'nudes',
            'cam', 'webcam', 'escort', 'hookup', 'dating',
            # Be comprehensive
            '18+', 'mature', 'explicit', 'uncensored',
        }

        # Video streaming keywords
        self.streaming_keywords = {
            'stream', 'watch', 'movie', 'film', 'series', 'tv show',
            'episode', 'season', 'cinema', 'hd', '4k', 'bluray',
            'download', 'torrent', 'subtitle', 'dubbed', 'subbed',
            # Piracy-specific
            'free movies', 'watch online', 'no signup', 'no ads',
            'latest movies', 'new releases', 'hollywood', 'bollywood',
        }

        # File hosting keywords
        self.filehost_keywords = {
            'upload', 'download', 'file sharing', 'cloud storage',
            'premium', 'debrid', 'cyberlocker', 'one-click',
            'direct link', 'mirror', 'backup',
        }

        # Social media keywords
        self.social_keywords = {
            'social network', 'follow', 'like', 'comment', 'share',
            'profile', 'friends', 'feed', 'timeline', 'post',
        }

    def _load_blocklists(self):
        """Load domain blocklists from files."""
        if self._blocklists_loaded:
            return

        blocklist_dir = self.blocklist_dir
        if not blocklist_dir.exists():
            # Blocklists not available, skip
            self._blocklists_loaded = True
            return

        # Load adult blocklists
        adult_files = [
            blocklist_dir / "adult_domains.txt",
            blocklist_dir / "ut1_adult.txt",
            blocklist_dir / "stevenblack_adult.txt",
        ]

        for filepath in adult_files:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            # Handle hosts file format (127.0.0.1 domain.com)
                            parts = line.split()
                            domain = parts[-1]  # Last part is the domain
                            self.adult_domains.add(domain.lower())

        # Load social media lists (optional)
        social_file = blocklist_dir / "social_domains.txt"
        if social_file.exists():
            with open(social_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        domain = parts[-1]
                        self.social_domains.add(domain.lower())

        self._blocklists_loaded = True

    def classify(
        self,
        domain: str,
        scraped_text: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ContentClassification:
        """
        Classify domain by content type.

        Args:
            domain: Domain to classify
            scraped_text: Optional scraped page text for keyword analysis
            metadata: Optional site metadata (language, cloudflare, etc.)

        Returns:
            ContentClassification with type, confidence, and reasons
        """
        self._load_blocklists()

        reasons = []
        scores = {
            ContentType.ADULT: 0.0,
            ContentType.VIDEO_STREAMING: 0.0,
            ContentType.FILE_HOSTING: 0.0,
            ContentType.EBOOK: 0.0,
            ContentType.SOCIAL: 0.0,
            ContentType.LEGITIMATE: 0.0,
        }

        domain_lower = domain.lower()

        # === Strategy 0: Hardcoded Blocklists from GTR Analysis (Highest Priority) ===

        if domain_lower in self.hardcoded_adult:
            scores[ContentType.ADULT] += 2.0  # Very high confidence
            reasons.append("hardcoded_adult_blocklist")

        if domain_lower in self.hardcoded_ebook:
            scores[ContentType.EBOOK] += 2.0  # Very high confidence
            reasons.append("hardcoded_ebook_blocklist")

        # Check social media platforms
        if domain_lower in self.hardcoded_social:
            scores[ContentType.SOCIAL] += 2.0  # Very high confidence
            reasons.append("hardcoded_social_platform")

        # Check legitimate sites (exact match first)
        if domain_lower in self.hardcoded_legitimate:
            scores[ContentType.LEGITIMATE] += 2.0  # Very high confidence
            reasons.append("hardcoded_legitimate_site")

        # Check for .edu and .gov domains
        if domain_lower.endswith('.edu') or '.edu.' in domain_lower:
            scores[ContentType.LEGITIMATE] += 2.0
            reasons.append("edu_domain")
        elif domain_lower.endswith('.gov') or '.gov.' in domain_lower:
            scores[ContentType.LEGITIMATE] += 2.0
            reasons.append("gov_domain")

        # === Strategy 1: File-based Blocklists (High Confidence) ===

        if domain_lower in self.adult_domains:
            scores[ContentType.ADULT] += 1.0
            reasons.append("adult_blocklist")

        if domain_lower in self.social_domains:
            scores[ContentType.SOCIAL] += 1.0
            reasons.append("social_blocklist")

        # === Strategy 2: Domain Pattern Matching (Medium Confidence) ===

        if self.adult_regex.search(domain_lower):
            scores[ContentType.ADULT] += 0.7
            reasons.append("adult_domain_pattern")

        if self.ebook_regex.search(domain_lower):
            scores[ContentType.EBOOK] += 0.8  # Higher confidence for ebook patterns
            reasons.append("ebook_domain_pattern")

        if self.filehost_regex.search(domain_lower):
            scores[ContentType.FILE_HOSTING] += 0.6
            reasons.append("filehost_domain_pattern")

        if self.streaming_regex.search(domain_lower):
            scores[ContentType.VIDEO_STREAMING] += 0.6
            reasons.append("streaming_domain_pattern")

        if self.social_regex.search(domain_lower):
            scores[ContentType.SOCIAL] += 0.6
            reasons.append("social_domain_pattern")

        # === Strategy 3: Content Keyword Analysis (Low-Medium Confidence) ===

        if scraped_text:
            text_lower = scraped_text.lower()

            # Count keyword matches
            adult_matches = sum(1 for kw in self.adult_keywords if kw in text_lower)
            streaming_matches = sum(1 for kw in self.streaming_keywords if kw in text_lower)
            filehost_matches = sum(1 for kw in self.filehost_keywords if kw in text_lower)
            social_matches = sum(1 for kw in self.social_keywords if kw in text_lower)

            # Threshold: Need at least 3 keyword matches
            if adult_matches >= 3:
                scores[ContentType.ADULT] += min(0.5, adult_matches * 0.1)
                reasons.append(f"adult_keywords_{adult_matches}")

            if streaming_matches >= 3:
                scores[ContentType.VIDEO_STREAMING] += min(0.5, streaming_matches * 0.1)
                reasons.append(f"streaming_keywords_{streaming_matches}")

            if filehost_matches >= 2:
                scores[ContentType.FILE_HOSTING] += min(0.4, filehost_matches * 0.1)
                reasons.append(f"filehost_keywords_{filehost_matches}")

            if social_matches >= 3:
                scores[ContentType.SOCIAL] += min(0.4, social_matches * 0.1)
                reasons.append(f"social_keywords_{social_matches}")

        # === Strategy 4: Metadata Analysis (Low Confidence) ===

        if metadata:
            # File hosting sites often have "premium" or "vip" in Telegram channels
            if metadata.get("has_premium_telegram"):
                scores[ContentType.FILE_HOSTING] += 0.2
                reasons.append("premium_telegram")

        # === Determine Final Classification ===

        # Get highest scoring category
        max_score = max(scores.values())

        if max_score == 0.0:
            # No classification possible
            return ContentClassification(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                reasons=["no_signals"],
                is_target=False,
            )

        # Get category with highest score
        content_type = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, max_score)

        # Normalize confidence to 0-1 range
        # Perfect match (blocklist) = 1.0
        # Multiple weak signals = 0.5-0.8
        # Single weak signal = 0.3-0.5
        if confidence > 1.0:
            confidence = 1.0

        # Both video_streaming AND file_hosting are targets
        # EXCLUDE: adult, ebook, social, legitimate
        is_target = (content_type in [ContentType.VIDEO_STREAMING, ContentType.FILE_HOSTING])

        return ContentClassification(
            content_type=content_type,
            confidence=confidence,
            reasons=reasons,
            is_target=is_target,
        )

    def is_adult_site(self, domain: str, scraped_text: Optional[str] = None) -> bool:
        """
        Quick check if domain is adult content.

        Args:
            domain: Domain to check
            scraped_text: Optional scraped text

        Returns:
            True if classified as adult content
        """
        classification = self.classify(domain, scraped_text)
        return classification.content_type == ContentType.ADULT

    def is_target_site(self, domain: str, scraped_text: Optional[str] = None) -> bool:
        """
        Check if domain is a video streaming piracy site (our target).

        Args:
            domain: Domain to check
            scraped_text: Optional scraped text

        Returns:
            True if classified as video streaming
        """
        classification = self.classify(domain, scraped_text)
        return classification.is_target

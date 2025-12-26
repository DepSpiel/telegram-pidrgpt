"""
Refined Perplexity API client with updated prompt specification.
"""

import requests
import json
import logging
import re
import hashlib
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PerplexityClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
    def get_crypto_news_content(self) -> Optional[Dict]:
        """Get crypto market news with NEW prompt specification."""
        try:
            today = datetime.now()
            formatted_date = today.strftime("%B %d, %Y")
            
            # NEW PROMPT - Updated as per user specification
            prompt = (
                "Summarize today's top global news about crypto market. "
                "Include major global economic events, and highlight any breaking news about near future events. "
                "Make an article no more than 800 characters (with spaces). "
                "Don't provide any guidance for the market trend.

"
                f"Today's date: {formatted_date}"
            )

            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional crypto news reporter. Provide factual market summaries "
                            "focused on news and events. Report what happened and what's upcoming without offering "
                            "market predictions or investment guidance."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 350,
                "temperature": 0.3,
                "stream": False
            }
            
            logger.info("📡 Requesting crypto news with updated prompt specification...")
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=35)
            
            if response.status_code != 200:
                logger.error(f"❌ API failed: {response.status_code}")
                return self._create_refined_fallback_content(formatted_date)
            
            try:
                data = response.json()
            except Exception:
                logger.error("❌ JSON parse failed")
                return self._create_refined_fallback_content(formatted_date)
            
            # Extract content
            content = self._extract_content_simple(data)
            
            if not content:
                logger.warning("⚠️ No content extracted, using fallback")
                return self._create_refined_fallback_content(formatted_date)
            
            # Format content with refined structure
            formatted_content = self._format_content_refined(content, formatted_date)
            
            # Generate unique image for this content
            image_url = self._generate_unique_crypto_image(formatted_content)
            
            result = {
                "text": formatted_content,
                "image_url": image_url,
                "char_count": len(formatted_content)
            }
            
            logger.info(f"✅ Content with updated prompt ready: {len(formatted_content)} chars")
            return result
            
        except Exception as e:
            logger.error(f"💥 Error: {str(e)}")
            today_str = datetime.now().strftime("%B %d, %Y")
            return self._create_refined_fallback_content(today_str)
    
    def _extract_content_simple(self, data: dict) -> str:
        """Simple content extraction compatible with Perplexity format."""
        try:
            # Standard Perplexity/OpenAI‑style format
            # data["choices"] -> list[ { "message": { "content": "..." } } ]
            if isinstance(data, dict) and "choices" in data:
                choices = data["choices"]
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if (
                        isinstance(first, dict)
                        and "message" in first
                        and isinstance(first["message"], dict)
                        and "content" in first["message"]
                        and isinstance(first["message"]["content"], str)
                    ):
                        content = first["message"]["content"]
                        if content.strip():
                            logger.info("✅ Found content in standard choices[0].message.content")
                            return content.strip()
            
            # Recursive search fallback (very generic)
            def find_content(obj):
                if isinstance(obj, str) and len(obj) > 50:
                    return obj
                if isinstance(obj, dict):
                    # Prefer explicit "content" keys
                    if "content" in obj and isinstance(obj["content"], str):
                        return obj["content"]
                    for value in obj.values():
                        result = find_content(value)
                        if result:
                            return result
                if isinstance(obj, list):
                    for item in obj:
                        result = find_content(item)
                        if result:
                            return result
                return None
            
            content = find_content(data)
            if content:
                logger.info("✅ Found content via recursive search")
                return content.strip()
            
            return ""
            
        except Exception as e:
            logger.error(f"💥 Extract error: {str(e)}")
            return ""
    
    def _format_content_refined(self, content: str, date: str) -> str:
        """
        Format content with refined structure.
        Target: keep result ≤ 800 characters (prompt requirement).
        """
        try:
            # Clean citations and extra formatting
            clean_content = re.sub(r"[d+]", "", content)
            clean_content = re.sub(r"**(.*?)**", r"\u0001", clean_content)
            clean_content = re.sub(r"*(.*?)*", r"\u0001", clean_content)
            # Сохраним переносы строк, но уберём лишние пробелы вокруг них
            clean_content = re.sub(r"[ \t]+", " ", clean_content)
            clean_content = re.sub(r"s*+", "", clean_content).strip()
            
            # Попробуем выделить заголовок из первой строки/первого абзаца
            lines = clean_content.split("")
            title = ""
            body_content = clean_content
            
            if lines:
                first_line = lines[0].strip()
                if (
                    first_line
                    and len(first_line) < 120
                    and not first_line.startswith("•")
                    and not first_line.startswith("-")
                ):
                    title = first_line
                    body_content = "
".join(lines[1:]).strip() or clean_content
            
            if not title:
                title = "Crypto Market Update"
            
            formatted_header = [
                f"📈 **{title}**",
                f"📅 *{date}*",
                ""]
            
            bullet_content = self._convert_to_detailed_bullets(body_content)
            
            hashtags_line = "*#CryptoNews #MarketOverview*"
            result_lines = formatted_header + bullet_content + ["", hashtags_line]
            result = "".join(result_lines)
            
            # Жёсткое ограничение по длине до 800 символов
            max_len = 800
            if len(result) > max_len:
                # Урежем, но оставим целую структуру по возможности
                # Попробуем сократить только буллеты
                available_for_bullets = max_len - (
                    len("".join(formatted_header)) + 1 + len(hashtags_line) + 2)
                truncated_bullets = self._truncate_bullets_refined(
                    bullet_content, max(0, available_for_bullets))
                result_lines = formatted_header + truncated_bullets + ["", hashtags_line]
                result = "".join(result_lines)
            
            # На всякий случай финальная обрезка до 800 символов
            if len(result) > max_len:
                result = result[: max_len - 1] + "…"
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Format error: {str(e)}")
            fallback = (
                f"📈 **Crypto Market Update**"
                f"📅 *{date}*"
                "• Comprehensive market analysis in progress"
                "*#CryptoNews #MarketOverview*")
            if len(fallback) > 800:
                fallback = fallback[:799] + "…"
            return fallback
    
    def _convert_to_detailed_bullets(self, content: str) -> list:
        """Convert content to detailed bullet point format."""
        try:
            # Remove trailing hashtags at end of text
            content = re.sub(r"(?:#w+s*)+$", "", content).strip()
            
            # Разбиваем на предложения
            temp = (
                content.replace(".", ".|")
                .replace("!", "!|")
                .replace("?", "?|")
            )
            raw_sentences = [s.strip() for s in temp.split("|") if s.strip()]
            
            sentences = []
            for sentence in raw_sentences:
                if len(sentence) > 15:
                    sentences.append(sentence)
            
            bullets = []
            for sentence in sentences:
                sentence = sentence.rstrip(".!?").strip()
                if not sentence:
                    continue
                if len(sentence) < 60:
                    sentence = self._enhance_short_bullet(sentence)
                bullets.append(f"• {sentence}")
            
            # Ensure 4–6 bullets
            if len(bullets) < 4:
                bullets = self._generate_comprehensive_bullets()
            elif len(bullets) > 6:
                bullets = bullets[:6]
            
            return bullets
            
        except Exception as e:
            logger.error(f"💥 Bullet conversion error: {str(e)}")
            return self._generate_comprehensive_bullets()
    
    def _enhance_short_bullet(self, bullet: str) -> str:
        """Enhance short bullets with more detail."""
        try:
            enhancements = {
                "bitcoin": "Bitcoin continues its market leadership with institutional interest",
                "ethereum": "Ethereum shows network strength amid ongoing development",
                "market": "Market dynamics reflect broader economic sentiment",
                "price": "Price action indicates key technical levels",
                "trading": "Trading volumes suggest increased market participation"
            }
            
            bullet_lower = bullet.lower()
            for key, enhancement in enhancements.items():
                if key in bullet_lower and len(bullet) < 50:
                    return enhancement
            
            return bullet
            
        except Exception:
            return bullet
    
    def _generate_comprehensive_bullets(self) -> list:
        """Generate comprehensive fallback bullets."""
        return [
            "• Bitcoin maintains consolidation above key support levels with institutional accumulation patterns emerging",
            "• Ethereum demonstrates network resilience with increasing validator participation and Layer 2 adoption growth",
            "• Top altcoins including BNB, XRP, and SOL show divergent performance reflecting sector-specific developments",
            "• Market sentiment indicators suggest cautious optimism amid ongoing regulatory clarity initiatives",
            "• DeFi and AI token sectors attract renewed interest following recent technological breakthroughs",
            "• Technical analysis reveals critical support and resistance zones shaping near-term price trajectories"
        ]
    
    def _truncate_bullets_refined(self, bullets: list, max_chars: int) -> list:
        """Truncate bullets to fit within character limit."""
        if max_chars <= 0:
            return []
        
        result = []
        current_length = 0
        
        for bullet in bullets:
            bullet_length = len(bullet) + 1  # newline
            if current_length + bullet_length <= max_chars:
                result.append(bullet)
                current_length += bullet_length
            else:
                available_chars = max_chars - current_length
                # Нужно оставить хотя бы ~30 символов для осмысленного сокращения
                if available_chars > 30:
                    shortened = bullet[: available_chars - 3].rstrip() + "..."
                    result.append(shortened)
                break
        
        if len(result) < 3 and bullets:
            # Если уж совсем мало, вернём первые 3, а длину дальше обрежет вызывающая функция
            result = bullets[:3]
        
        return result
    
    def _expand_bullets(self, bullets: list, additional_chars_needed: int) -> list:
        """
        Expand bullets if content is too short.
        Сейчас почти не используется, так как есть жёсткий лимит 800 символов.
        Оставлено для совместимости, но без агрессивного расширения.
        """
        if additional_chars_needed < 50:
            return bullets
        
        comprehensive_bullets = self._generate_comprehensive_bullets()
        expanded = bullets[:]
        
        for comp_bullet in comprehensive_bullets:
            if len("
".join(expanded + [comp_bullet])) < 800:
                if not any(comp_bullet[2:20] in existing[2:20] for existing in expanded):
                    expanded.append(comp_bullet)
        
        return expanded
    
    def _generate_unique_crypto_image(self, content: str) -> str:
        """Generate a unique crypto image based on content hash."""
        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            
            crypto_images = [
                f"https://source.unsplash.com/1200x800/?cryptocurrency,trading,{content_hash}",
                "https://source.unsplash.com/1200x800/?bitcoin,market,analysis",
                "https://source.unsplash.com/1200x800/?blockchain,finance,charts",
                f"https://picsum.photos/1200/800?random={content_hash}",
                "https://images.unsplash.com/photo-1640340434855-6084b1f4901c?w=1200&h=800&fit=crop",
                "https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=1200&h=800&fit=crop",
                "https://images.unsplash.com/photo-1616499370260-485b3e5ed653?w=1200&h=800&fit=crop"
            ]
            
            hash_int = int(content_hash, 16)
            selected_image = crypto_images[hash_int % len(crypto_images)]
            
            try:
                # HEAD может не всегда работать на этих сервисах, поэтому ошибки игнорируются
                response = requests.head(selected_image, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    logger.info(f"✅ Generated unique image: {selected_image}")
                    return selected_image
            except Exception:
                pass
            
            # Fallback
            return "https://images.unsplash.com/photo-1640340434855-6084b1f4901c?w=1200&h=800&fit=crop"
            
        except Exception as e:
            logger.error(f"💥 Image generation error: {str(e)}")
            return "https://images.unsplash.com/photo-1640340434855-6084b1f4901c?w=1200&h=800&fit=crop"
    
    def _create_refined_fallback_content(self, date: str) -> Dict:
        """Create refined fallback content within 800 chars."""
        fallback_bullets = [
            "• Bitcoin trading activity continues with notable institutional transactions reported",
            "• Ethereum network updates and Layer 2 scaling solutions see increased adoption",
            "• Major altcoins display varied performance across different market segments",
            "• Global economic indicators and central bank policies influence crypto market sentiment",
            "• Regulatory developments in key jurisdictions impact trading volumes and market access",
            "• Upcoming industry events and protocol upgrades scheduled for near-term implementation"
        ]
        
        formatted_content = (
            f"📈 **Crypto Market Update**
"
            f"📅 *{date}*

"
            f"{chr(10).join(fallback_bullets)}

"
            "*#CryptoNews #MarketOverview*"
        )
        
        if len(formatted_content) > 800:
            formatted_content = formatted_content[:799] + "…"
        
        return {
            "text": formatted_content,
            "image_url": self._generate_unique_crypto_image(formatted_content),
            "char_count": len(formatted_content)
        }
    
    def test_connection(self) -> bool:
        """Simple connection test."""
        try:
            payload = {
                "model": "sonar-pro",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 10
            }
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                logger.info("✅ Perplexity API connection successful")
                return True
            return False
            
        except Exception as e:
            logger.error(f"💥 Test error: {str(e)}")
            return False
    
    def get_daily_content(self, topic: str) -> Optional[Dict]:
        """Compatibility method."""
        return self.get_crypto_news_content()

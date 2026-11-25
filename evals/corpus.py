"""记忆召回评测语料：persona 记忆库 + 带 gold 标注的查询。

为什么不用模板批量生成
----------------------
模板生成的 query 会复用记忆原文的实词，词法召回必然虚高，测出来的分数
不能代表真实用户体验。这里的 query 全部手写，并且**刻意制造词汇鸿沟**：
记忆写「只喝美式不加糖不加奶」，用户问「对咖啡有什么讲究」；记忆写
「讨厌开长会」，用户问「最烦什么类型的工作安排」。字面重叠越少，越接近
真实提问，也越能暴露纯词法系统的真实上限。

为什么要有 stash / 干扰项
-------------------------
所有系统在「我叫什么名字」上都能满分，区分度全在易混条目上。所以每个
persona 都刻意埋了同实体不同值、同数字不同事、相近表述不同人的记忆。

字段说明
--------
id      语义化主键，query 的 gold 引用的就是它。语义化而非 a01 这类编号，
        是为了让「知识更新」两组关系在数据里直接可读（old / new）。
kind    记忆类型，participates in KIND_WEIGHT 加权。
ago     距今多少天写入。时间衰减半衰期 180 天，所以这个值直接影响排序。
sess    写入批次（同一轮对话）。邻域扩展按 session 分组取邻居。
text    记忆正文，一句话陈述事实，不写成对话。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 2026-11-25 20:00 (CST)。固定值，所有时间计算都相对它，保证结果可复现。
NOW = 1795608000.0


@dataclass(frozen=True)
class Mem:
    id: str
    kind: str
    ts: float
    sess: str
    text: str
    # 写入侧问法扩展（index-time expansion）。生产环境应由模型在落库时生成，
    # 这里由 expand.py 用受控词表模拟，用于量化「扩写值多少 pp」。
    aliases: tuple[str, ...] = ()
    # 被哪条记忆取代。知识更新题里召回到 old 算错，不能只算「没召回」。
    superseded_by: str | None = None


@dataclass(frozen=True)
class Query:
    qid: str
    persona: str
    bucket: str
    text: str
    gold: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    note: str = ""


def _mk(persona: str, rows: list[tuple]) -> list[Mem]:
    out: list[Mem] = []
    for mid, kind, ago, sess, text in rows:
        out.append(Mem(id=mid, kind=kind, ts=NOW - float(ago) * 86400.0,
                       sess=sess, text=text))
    return out


# ============================================================================
# Persona A：28 岁后端工程师，深圳。技术日常 + 亲密关系 + 健康
# ============================================================================
_A = _mk("A", [
    ("a_name", "profile", 380, "s0", "用户叫陈屿，28 岁，在深圳一家 SaaS 公司做后端开发，主语言是 Python"),
    ("a_company", "profile", 380, "s0", "用户所在的公司叫云栈科技，做企业报销系统，团队二十人左右"),
    ("a_gf", "profile", 360, "s1", "用户的女朋友叫小雅，在一家广告公司做平面设计，比他小两岁"),
    ("a_cat", "profile", 355, "s1", "用户养了一只橘猫，名字叫团子，今年四岁"),
    ("a_parents", "profile", 350, "s1", "用户父母住在湖南长沙，父亲退休前是中学物理老师"),
    ("a_coffee", "preference", 340, "s2", "用户喝咖啡只喝美式，不加糖也不加奶，觉得加奶是浪费豆子"),
    ("a_meeting", "preference", 338, "s2", "用户非常讨厌开长会，超过半小时的会他会明显烦躁"),
    ("a_theme", "preference", 335, "s2", "用户所有编辑器和终端都用深色主题，说亮色看久了眼睛疼"),
    ("a_cilantro", "preference", 330, "s2", "用户吃不了香菜，闻到味道就难受，点外卖每次都要备注"),
    ("a_brevity", "preference", 328, "s2", "用户喜欢简洁直接的回复，讨厌铺垫和长篇大论"),
    ("a_answer_style", "preference", 300, "s3", "用户要求助手讲技术问题时先给结论再给理由，不要绕圈子"),
    ("a_nickname", "preference", 295, "s3", "用户不喜欢被人叫陈总，觉得别扭，直接叫名字就行"),
    ("a_code_mark", "preference", 290, "s3", "用户要求助手发代码时带上语言标记，方便直接复制"),
    ("a_chengdu", "event", 250, "s4", "今年三月中旬用户去成都出差了五天，住在春熙路附近"),
    ("a_overtime", "event", 34, "s5", "上个月公司项目上线，用户连续加班到凌晨两点，第二天请假了半天"),
    ("a_vaccine", "event", 15, "s5", "上上周用户去打了第三针疫苗，胳膊疼了两天"),
    ("a_share", "event", 9, "s6", "上周用户在公司内部分享了异步 IO 的技术方案，讲了三十分钟"),
    ("a_quarrel", "event", 40, "s6", "上个月用户和小雅因为租房的事吵了一架，第二天和好了"),
    ("a_cat_sick", "event", 18, "s7", "半个月前团子吐了毛球，用户带它去宠物医院，花了八百多"),
    ("a_changsha", "event", 400, "s7", "去年十一用户回了趟长沙老家，待了四天"),
    ("a_macos", "event", 1, "s8", "昨天用户的 MacBook 系统更新到了最新版本，卡了一下午"),
    ("a_rust", "fact", 120, "s3", "用户正在自学 Rust，看到所有权和借用那一章，觉得生命周期最难"),
    ("a_laptop", "fact", 300, "s3", "用户用的电脑是 MacBook Pro M3，16G 内存，硬盘 512G"),
    ("a_editor_old", "fact", 320, "s3", "用户一直用 VSCode 写代码，装了一堆插件"),
    ("a_editor_new", "fact", 26, "s6", "用户把编辑器从 VSCode 换成了 Cursor，说 AI 补全好用很多"),
    ("a_phone", "fact", 280, "s3", "用户的手机是 iPhone 15，用了两年，电池健康只剩 79%"),
    ("a_blog_domain", "fact", 200, "s4", "用户的博客域名是 chenyu.dev，DNS 挂在 Cloudflare 上"),
    ("a_router", "fact", 180, "s4", "用户家里是千兆宽带，路由器是小米 AX6000，放在客厅电视柜上"),
    ("a_wedding", "fact", 60, "s6", "用户和小雅计划 2027 年 5 月 20 日领证，正在看房"),
    ("a_birthday", "fact", 375, "s0", "用户的生日是 3 月 12 日，双鱼座"),
    ("a_japanese", "fact", 45, "s6", "用户在用多邻国学日语，已经连续打卡四十天，每天二十分钟"),
    ("a_regex", "fact", 150, "s4", "用户不太会写正则，每次都要现查语法"),
    ("a_liver", "fact", 100, "s5", "用户体检有轻度脂肪肝，医生让他少喝酒、每周运动三次"),
    ("a_github", "fact", 220, "s4", "用户的 GitHub 用户名是 chenyu-dev"),
    ("a_game", "preference", 130, "s5", "用户周末喜欢在家打艾尔登法环，玩了一年多还没通关"),
    ("a_job_plan", "plan", 50, "s6", "用户打算在年底前后换一份工作，目标是大厂或者全远程的岗位"),
    ("a_japan", "plan", 48, "s6", "用户想明年春天带小雅去日本玩一趟，预算三万，正在攒钱"),
    ("a_astro", "plan", 30, "s7", "用户打算把博客从 Hexo 迁移到 Astro，嫌 Hexo 构建太慢"),
    ("a_water_reminder", "agreement", 70, "s6", "用户和助手约定，每天晚上十点提醒他喝一杯水"),
    ("a_ssl", "self", 12, "s7", "助手提醒过用户，他的 SSL 证书 12 月 3 日到期，需要续期"),
    ("a_txt", "self", 45, "s6", "助手帮用户在桌面创建了 test.txt，里面写入了 1234"),
    ("a_quicksort", "self", 8, "s8", "助手上周给用户写过一段快速排序的 Python 实现，用了双指针分区"),
])

# ============================================================================
# Persona B：34 岁会计，成都。育儿 + 家庭财务 + 过敏等安全信息
# ============================================================================
_B = _mk("B", [
    ("b_name", "profile", 300, "s0", "用户叫林蔓，34 岁，在成都一家建材公司做会计，月底最忙"),
    ("b_kid", "profile", 298, "s0", "用户的女儿叫朵朵，今年七岁，刚上小学一年级"),
    ("b_kid_old", "fact", 290, "s0", "朵朵之前一直在上幼儿园大班，老师说她性格外向"),
    ("b_husband", "profile", 288, "s0", "用户的丈夫叫何伟，在一家物流公司跑调度，经常上夜班"),
    ("b_allergy", "fact", 285, "s1", "朵朵对花生严重过敏，碰一点就会起疹子，家里从来不放花生油"),
    ("b_allergy_mom", "fact", 283, "s1", "用户自己对芒果过敏，但症状轻微，只是嘴周发痒"),
    ("b_health", "fact", 200, "s2", "用户有甲状腺结节，每半年要复查一次，目前是良性"),
    ("b_house", "plan", 260, "s1", "用户家在攒钱买第二套房，目标总价两百万以内，首付还差三十万"),
    ("b_piano", "plan", 180, "s3", "用户想让朵朵学钢琴，看中了家附近一家机构，一节课两百"),
    ("b_work_busy", "fact", 170, "s3", "用户月底结账那几天经常要加班到九点，回家朵朵已经睡了"),
    ("b_cooking", "preference", 160, "s4", "用户做饭口味清淡，几乎不放辣椒，但丈夫爱吃辣，会单独炒一份"),
    ("b_sleep", "preference", 150, "s4", "用户睡眠浅，丈夫上夜班回来开门会把她吵醒，后来换了静音门锁"),
    ("b_park", "preference", 140, "s4", "用户周末喜欢带朵朵去人民公园喂鸽子，那边有家冰粉很好吃"),
    ("b_temple", "event", 120, "s5", "今年四月用户带朵朵去了趟峨眉山，爬到半山腰朵朵就走不动了"),
    ("b_school", "event", 95, "s5", "九月开学那天用户请假送朵朵去学校，朵朵在校门口哭了一场"),
    ("b_train", "event", 60, "s6", "上个月用户坐高铁去了趟重庆，当天来回，是为了办事"),
    ("b_birthday_kid", "fact", 250, "s1", "朵朵的生日是 6 月 1 日，正好是儿童节"),
    ("b_birthday_husband", "fact", 240, "s1", "何伟的生日是 11 月 8 日，用户每年都给他买剃须刀"),
    ("b_tuition", "fact", 130, "s5", "朵朵学校这学期课后延时服务费是八百块，一次性交"),
    ("b_phone", "fact", 220, "s2", "用户用的手机是红米，屏幕碎过一次，换过外屏"),
    ("b_boss", "fact", 210, "s2", "用户的直属领导是财务总监姓周，要求所有报销单必须贴发票原件"),
    ("b_bank", "fact", 190, "s3", "用户的工资卡是招商银行的，每月十号发工资"),
    ("b_mom", "fact", 185, "s3", "用户的妈妈住在成都郊区，每周三会过来帮忙带朵朵"),
    ("b_english", "fact", 100, "s5", "朵朵在学英语自然拼读，每周二四晚上有网课"),
    ("b_cat", "fact", 175, "s3", "用户家养了一只英短蓝猫，叫汤圆，特别能吃"),
    ("b_migraine", "event", 30, "s6", "用户上周末偏头痛发作，躺了一整天，什么都没干"),
    ("b_exam", "plan", 45, "s6", "用户准备今年考中级会计职称，十月报名，已经在看书了"),
    ("b_travel", "plan", 40, "s6", "用户计划寒假带朵朵去三亚看海，怕冷，想找暖和的地方"),
    ("b_agreement", "agreement", 55, "s6", "用户让助手每个月最后一天提醒她核对报销单据，别漏了"),
    ("b_self_excel", "self", 35, "s6", "助手帮用户写过一段 Excel 公式，用 SUMIFS 按部门汇总差旅费"),
    ("b_self_letter", "self", 20, "s7", "助手帮用户起草过一封给老师的请假条，理由是朵朵发烧"),
    ("b_nosugar", "preference", 110, "s5", "用户不爱喝甜饮料，奶茶只点三分糖或者无糖"),
    ("b_shoes", "event", 25, "s6", "用户给朵朵买了双新运动鞋，36 码，结果买大了一号"),
    ("b_rent", "fact", 165, "s4", "用户现在住的房子是租的，两室一厅，月租三千二"),
])

# ============================================================================
# Persona C：67 岁退休老人，哈尔滨。健康用药 + 孙辈 + 设备能力受限
# ============================================================================
_C = _mk("C", [
    ("c_name", "profile", 260, "s0", "用户叫王守义，六十七岁，退休前在哈尔滨一家锅炉厂做技术员"),
    ("c_wife", "profile", 258, "s0", "用户的老伴叫张桂兰，两个人结婚四十三年了"),
    ("c_grand", "profile", 255, "s0", "用户的孙女叫王悦，在北京读大学，学的是临床医学"),
    ("c_bp", "fact", 250, "s1", "用户有高血压，高压常年在一百六左右，每天早晚各量一次"),
    ("c_med", "fact", 248, "s1", "用户每天早饭后吃一片苯磺酸氨氯地平，不能断，断了会头晕"),
    ("c_med2", "fact", 246, "s1", "用户还在吃阿司匹林肠溶片，一天一片，饭后吃"),
    ("c_knee", "fact", 240, "s1", "用户的左膝盖有骨刺，上下楼梯疼，医生建议少爬山"),
    ("c_fall", "event", 300, "s1", "去年冬天用户在小区门口滑倒过一次，摔到了尾椎，躺了半个月"),
    ("c_phone", "fact", 230, "s2", "用户用的是一部老年机，字很大，只会接打电话和看短信"),
    ("c_wechat", "fact", 228, "s2", "用户的孙女给他装过微信，但他只会点开看，不会打字，都发语音"),
    ("c_chess", "preference", 220, "s3", "用户每天下午去小区凉亭下象棋，最爱用中炮开局"),
    ("c_flower", "preference", 218, "s3", "用户在阳台上养了六盆君子兰，每年冬天下雪前要搬进屋"),
    ("c_food", "preference", 215, "s3", "用户口味重，爱吃咸的，但医生让少盐，老伴做菜越来越淡，他总抱怨"),
    ("c_alcohol", "preference", 210, "s3", "用户以前爱喝白酒，现在被老伴管着，一顿最多一两"),
    ("c_style", "preference", 205, "s4", "用户看字费劲，希望回答短一点、字大一点，别用生词"),
    ("c_hearing", "fact", 200, "s4", "用户右耳听力下降，别人说话声音小了听不清，电视音量总开很大"),
    ("c_pension", "fact", 195, "s4", "用户的退休金每月四千二，每月十五号到账"),
    ("c_son", "fact", 190, "s4", "用户的儿子叫王建军，在深圳做装修，一年回来两三次"),
    ("c_newyear", "event", 330, "s2", "今年过年儿子一家回来住了六天，孙女在家包了饺子"),
    ("c_hospital", "event", 90, "s5", "上个月用户去医院复查血压，医生把药量调小了半片"),
    ("c_cataract", "event", 70, "s5", "九月用户做了白内障手术，是左眼，恢复得不错"),
    ("c_trip", "event", 150, "s4", "今年夏天用户和老伴跟团去了趟北戴河，坐大巴坐了十个小时"),
    ("c_boiler", "fact", 180, "s4", "用户家是老小区，集中供暖，每年十月二十号左右试水"),
    ("c_neighbor", "fact", 170, "s5", "用户对门住着老李，两人经常一起遛弯，老李有糖尿病"),
    ("c_dog", "fact", 160, "s5", "用户家养了一条小土狗，叫大黄，是孙女起的名"),
    ("c_cooking", "preference", 155, "s5", "用户爱吃面食，老伴每周包一次酸菜馅饺子"),
    ("c_agreement", "agreement", 100, "s5", "用户让助手每天提醒他晚上八点吃降压药，别忘了"),
    ("c_self_weather", "self", 40, "s6", "助手帮用户查过哈尔滨的天气，说那几天要降温到零下二十度"),
    ("c_self_sms", "self", 35, "s6", "助手帮用户念过一条短信，是社保局发的认证通知"),
    ("c_sleep", "fact", 145, "s5", "用户睡眠不好，凌晨三四点常醒，白天要补一觉"),
])

# ============================================================================
# Persona D：31 岁创业者，杭州。工作压力 + 团队 + 融资，信息更新频繁
# ============================================================================
_D = _mk("D", [
    ("d_name", "profile", 240, "s0", "用户叫苏黎，31 岁，在杭州做跨境电商，公司叫泊远"),
    ("d_team", "fact", 238, "s0", "用户公司现在十二个人，其中运营六个，开发三个，设计两个，加她自己"),
    ("d_partner", "profile", 235, "s0", "用户的合伙人是老周，负责供应链，两人认识八年了"),
    ("d_biz", "fact", 230, "s1", "用户公司主营家居小件，主要卖到北美，亚马逊和独立站各占一半"),
    ("d_round_old", "fact", 200, "s1", "用户年初在谈天使轮，目标五百万，估值三千万"),
    ("d_round_new", "fact", 35, "s6", "用户的天使轮已经close了，实际拿到三百五十万，估值两千八百万"),
    ("d_investor", "fact", 60, "s5", "领投的机构叫启元资本，投过三家出海品牌"),
    ("d_office", "fact", 180, "s2", "用户办公室在滨江，三百平，租金一个月两万八"),
    ("d_insomnia", "fact", 90, "s4", "用户长期失眠，要听着白噪音才能睡着，经常凌晨两点还醒着"),
    ("d_anxiety", "fact", 88, "s4", "用户压力大的时候会胃疼，去年查出浅表性胃炎"),
    ("d_home", "fact", 175, "s2", "用户在杭州临平买了房，一个人住，九十平，两只布偶猫"),
    ("d_cat", "profile", 174, "s2", "用户的两只布偶猫叫元宝和发财，都是母的"),
    ("d_coffee", "preference", 170, "s3", "用户一天要喝三杯手冲，偏爱浅烘的耶加雪菲"),
    ("d_drink", "preference", 168, "s3", "用户应酬时喝不了高度酒，一杯就脸红，一般用红酒应付"),
    ("d_style", "preference", 165, "s3", "用户喜欢看表格和要点，讨厌大段文字，越结构化越好"),
    ("d_agreement_weekly", "agreement", 120, "s4", "用户让助手每周一早上给她一份上周待办回顾，按优先级排"),
    ("d_agreement_meeting", "agreement", 118, "s4", "用户要求助手在会议纪要里必须单列「待办」和「负责人」两栏"),
    ("d_self_bp", "self", 50, "s5", "助手帮用户写过一个商业计划书的目录框架，分了七章"),
    ("d_self_pricing", "self", 28, "s6", "助手帮用户算过一款收纳盒的定价，按成本加价三倍算毛利率是 62%"),
    ("d_supplier", "fact", 140, "s3", "用户的主力供应商在义乌，姓吴，合作三年，账期四十五天"),
    ("d_logistics", "fact", 130, "s3", "用户走的是海外仓模式，美国仓在洛杉矶，头程走海运"),
    ("d_hire", "plan", 80, "s4", "用户想招一个懂 Google 广告投放的人，预算月薪两万五"),
    ("d_expand", "plan", 75, "s4", "用户计划明年进欧洲市场，先做德国，因为 VAT 相对好处理"),
    ("d_event_listing", "event", 110, "s5", "上个月用户的一个爆款被跟卖了，销量掉了四成"),
    ("d_event_fire", "event", 100, "s5", "十月用户团队有个运营离职，走得很突然，交接只做了半天"),
    ("d_event_trip", "event", 65, "s5", "上上个月用户去深圳参加了一个跨境展会，待了三天"),
    ("d_event_lawsuit", "event", 20, "s6", "用户公司上个月被一家美国律所发了侵权警告函，涉及一款杯垫图案"),
    ("d_birthday", "fact", 160, "s3", "用户的生日是 9 月 9 日"),
    ("d_sleep_med", "fact", 85, "s4", "用户试过褪黑素，效果一般，后来改成睡前不看手机"),
])

# ============================================================================
# Persona E：21 岁大三学生，武汉。学业 + 考研 + 异地恋 + 兼职
# ============================================================================
_E = _mk("E", [
    ("e_name", "profile", 200, "s0", "用户叫李知行，21 岁，武汉一所 211 读计算机，现在大三"),
    ("e_gf", "profile", 198, "s0", "用户的女朋友叫周雨，在西安读书，两人异地，一般两周见一次"),
    ("e_roommate", "fact", 195, "s0", "用户宿舍四个人，室友分别叫大壮、阿凯、老谭"),
    ("e_exam_plan", "plan", 190, "s1", "用户准备考研，目标是本校的计算机专硕，专业课考 408"),
    ("e_exam_school", "fact", 188, "s1", "用户考研想冲一下华科的计算机，但觉得分数线太高，还在犹豫"),
    ("e_study_time", "fact", 120, "s2", "用户每天在图书馆待到闭馆，晚上十点半回宿舍"),
    ("e_gpa", "fact", 180, "s1", "用户的绩点是 3.6，专业排名二十名左右，保研希望不大"),
    ("e_parttime", "fact", 150, "s2", "用户在做家教兼职，教一个初二学生数学，一小时一百二"),
    ("e_money", "fact", 145, "s2", "用户每个月生活费两千，家教能挣八百，基本够花"),
    ("e_game", "preference", 140, "s3", "用户爱打无畏契约，主玩狙击位，段位是钻石"),
    ("e_anime", "preference", 138, "s3", "用户喜欢看番，最近在追一部讲乐队的动画"),
    ("e_spicy", "preference", 135, "s3", "用户能吃辣，武汉的热干面要吃加辣的，鸭脖一次能吃一斤"),
    ("e_night", "fact", 130, "s3", "用户习惯熬夜，一般凌晨一点睡，早上八点起，靠咖啡撑着"),
    ("e_cat_family", "fact", 125, "s3", "用户老家在河南信阳，家里养了一条柯基叫豆豆"),
    ("e_dog_sick", "event", 30, "s5", "上个月豆豆不吃东西，家里人带去医院，说是肠胃炎，打了三天针"),
    ("e_trip_xian", "event", 55, "s4", "国庆用户去了西安找周雨，一起去了大唐不夜城，人挤得走不动"),
    ("e_trip_budget", "fact", 50, "s4", "用户那次去西安花了大概一千五，其中高铁票六百"),
    ("e_contest", "event", 80, "s4", "用户上半年参加蓝桥杯，拿了个省二，差一点进国赛"),
    ("e_project", "fact", 100, "s4", "用户在做一个基于 SpringBoot 的二手书交易平台，作为课程设计"),
    ("e_project_block", "event", 25, "s5", "用户的课程设计卡在支付接口，沙箱环境一直回调失败"),
    ("e_roommate_conflict", "event", 40, "s5", "用户和室友大壮因为晚上打游戏声音大吵过一次"),
    ("e_sleep_issue", "fact", 110, "s4", "用户宿舍晚上断电后网就断了，只能开热点"),
    ("e_english", "fact", 105, "s4", "用户英语六级考了两次，第二次 452 分过了"),
    ("e_agreement", "agreement", 90, "s4", "用户让助手每周日晚上帮他列一下下周的复习计划"),
    ("e_self_sort", "self", 35, "s5", "助手帮用户讲过快速排序和归并排序的区别，还画了示意图"),
    ("e_self_resume", "self", 22, "s5", "助手帮用户改过简历，把课程设计那段改成了 STAR 结构"),
    ("e_self_bug", "self", 12, "s6", "助手帮用户排查过一个空指针异常，原因是实体类字段名和数据库不一致"),
])

ALL_MEMORIES: list[Mem] = [*_A, *_B, *_C, *_D, *_E]

# 知识更新关系：召回到 old 而不召回 new，是「答错」而不是「没答」
_SUPERSEDE = {
    "a_editor_old": "a_editor_new",
    "b_kid_old": "b_kid",
    "d_round_old": "d_round_new",
}
# Mem 是 frozen dataclass，不能原地改字段，用重新构造的方式打上 superseded_by
ALL_MEMORIES = [
    Mem(id=m.id, kind=m.kind, ts=m.ts, sess=m.sess, text=m.text,
        aliases=m.aliases, superseded_by=_SUPERSEDE.get(m.id))
    for m in ALL_MEMORIES
]
_MEM_BY_ID: dict[str, Mem] = {m.id: m for m in ALL_MEMORIES}


# ============================================================================
# 查询集
# ============================================================================
# bucket 取值与含义：
#   easy        字面基本重合，考「有没有存」
#   gap         词汇鸿沟：同义 / 上下位 / 口语化，考「换个说法还认不认得」
#   anaphora    指代与省略：「那个」「上次说的」，中文高频
#   temporal    时间指代：「上个月」「去年冬天」
#   update      知识更新：有旧值干扰，召回旧的算错
#   multihop    需要两条以上组合
#   distractor  有强干扰项：同实体不同值 / 同数字不同事
#   abstain     库里没有，正确行为是不注入
#   self        机器人的自我记忆：它说过什么、做过什么、承诺过什么
#   agreement   双方约定：需要按约定行事

QUERIES: list[Query] = [
    # ---------------- Persona A ----------------
    Query("a_q01", "A", "easy", "我叫什么名字，多大了", ("a_name",)),
    Query("a_q02", "A", "easy", "我家猫叫什么", ("a_cat",)),
    Query("a_q03", "A", "easy", "我女朋友是做什么工作的", ("a_gf",)),
    Query("a_q04", "A", "gap", "我对咖啡有什么讲究", ("a_coffee",),
          note="记忆写「只喝美式不加糖不加奶」，query 只共享「咖啡」"),
    Query("a_q05", "A", "gap", "我最烦什么类型的工作安排", ("a_meeting",),
          note="记忆写「讨厌开长会」，query 无「会」字"),
    Query("a_q06", "A", "anaphora", "那个补全特别好用的，我现在一直在用", ("a_editor_new",)),
    Query("a_q07", "A", "temporal", "上个月加班最狠是哪次", ("a_overtime",)),
    Query("a_q08", "A", "multihop", "我打算换工作，想找什么类型的公司", ("a_job_plan",)),
    Query("a_q09", "A", "update", "我现在用哪个编辑器写代码", ("a_editor_new",), ("a_editor_old",)),
    Query("a_q10", "A", "abstain", "我妹妹今年多大", note="用户没有妹妹"),
    Query("a_q11", "A", "distractor", "我生日是哪天", ("a_birthday",),
          note="干扰：a_wedding 含 5 月 20 日，同为日期类"),
    Query("a_q12", "A", "gap", "我身体上有什么要注意的", ("a_liver",)),
    Query("a_q13", "A", "self", "你之前说我的证书什么时候到期", ("a_ssl",)),
    Query("a_q14", "A", "self", "你在桌面给我建过什么文件", ("a_txt",)),
    Query("a_q15", "A", "agreement", "我们之前说好每天晚上干什么", ("a_water_reminder",)),
    Query("a_q16", "A", "gap", "你回我消息要注意什么风格", ("a_answer_style", "a_brevity")),
    Query("a_q17", "A", "temporal", "我日语打卡多久了", ("a_japanese",)),
    Query("a_q18", "A", "gap", "我不吃什么东西", ("a_cilantro",)),
    Query("a_q19", "A", "multihop", "我明年春天想出去玩，预算多少", ("a_japan",)),
    Query("a_q20", "A", "distractor", "我养了什么宠物，上次生病花了多少钱", ("a_cat", "a_cat_sick")),

    # ---------------- Persona B ----------------
    Query("b_q01", "B", "easy", "我女儿叫什么，几岁了", ("b_kid",)),
    Query("b_q02", "B", "easy", "我在哪家公司上班", ("b_name",)),
    Query("b_q03", "B", "gap", "朵朵吃东西有什么禁忌", ("b_allergy",),
          note="记忆写「对花生严重过敏」，query 用「禁忌」"),
    Query("b_q04", "B", "distractor", "我女儿生日是哪天", ("b_birthday_kid",),
          note="干扰：b_birthday_husband 也是生日"),
    Query("b_q05", "B", "update", "朵朵现在上学了吗", ("b_kid",), ("b_kid_old",)),
    Query("b_q06", "B", "temporal", "我上个月去重庆干嘛了", ("b_train",)),
    Query("b_q07", "B", "gap", "我做饭口味怎么样", ("b_cooking",)),
    Query("b_q08", "B", "multihop", "我想给朵朵报个兴趣班，之前看中的那个多少钱", ("b_piano",)),
    Query("b_q09", "B", "gap", "我最近睡眠有什么问题", ("b_sleep",)),
    Query("b_q10", "B", "self", "你之前帮我写过 Excel 公式吗", ("b_self_excel",)),
    Query("b_q11", "B", "agreement", "月底你要提醒我做什么", ("b_agreement",)),
    Query("b_q12", "B", "abstain", "我儿子的成绩怎么样", note="用户只有一个女儿"),
    Query("b_q13", "B", "distractor", "我对什么过敏", ("b_allergy_mom",),
          note="干扰：女儿对花生过敏、自己对芒果过敏，两处都写「过敏」"),
    Query("b_q14", "B", "gap", "我身体有什么老毛病要复查", ("b_health",)),
    Query("b_q15", "B", "multihop", "我家攒钱买房还差多少", ("b_house",)),
    Query("b_q16", "B", "temporal", "我今年考什么证", ("b_exam",)),
    Query("b_q17", "B", "easy", "我家猫叫什么", ("b_cat",)),
    Query("b_q18", "B", "gap", "我喝奶茶一般怎么点", ("b_nosugar",)),

    # ---------------- Persona C ----------------
    Query("c_q01", "C", "easy", "我叫什么，以前是做什么的", ("c_name",)),
    Query("c_q02", "C", "gap", "我每天要吃哪些药", ("c_med", "c_med2"),
          note="两条都是用药，必须都召回才安全"),
    Query("c_q03", "C", "distractor", "我血压高不高", ("c_bp",),
          note="干扰：c_hospital 也提血压，但说的是复查调药"),
    Query("c_q04", "C", "temporal", "我去年冬天摔的那次是怎么回事", ("c_fall",)),
    Query("c_q05", "C", "gap", "我腿脚有什么毛病", ("c_knee",)),
    Query("c_q06", "C", "self", "你帮我查过什么天气", ("c_self_weather",)),
    Query("c_q07", "C", "agreement", "我让你每天几点提醒我吃药", ("c_agreement",)),
    # 「孙子」是陷阱：用户只有孙女。正确行为是召回孙女那条并纠正，所以
    # gold 是 c_grand（否则模型无从得知"没有孙子"），但不属于 abstain。
    Query("c_q08", "C", "distractor", "我孙子今年考上大学了吗", ("c_grand",),
          note="用户只有孙女，问孙子应先召回孙女再纠正"),
    Query("c_q09", "C", "gap", "我用手机有什么困难", ("c_phone", "c_wechat")),
    Query("c_q10", "C", "easy", "我孙女在哪读书", ("c_grand",)),
    Query("c_q11", "C", "temporal", "我什么时候做的眼睛手术", ("c_cataract",)),
    Query("c_q12", "C", "gap", "我平时有什么爱好", ("c_chess", "c_flower")),
    Query("c_q13", "C", "multihop", "我退休金多少，够不够花", ("c_pension",)),
    Query("c_q14", "C", "distractor", "我家那条狗叫什么", ("c_dog",),
          note="干扰：其他 persona 也有宠物，跨 persona 不应召回"),
    Query("c_q15", "C", "gap", "我耳朵不太好吧", ("c_hearing",)),

    # ---------------- Persona D ----------------
    Query("d_q01", "D", "easy", "我公司叫什么，做什么的", ("d_name", "d_biz")),
    Query("d_q02", "D", "update", "我融资现在是什么情况", ("d_round_new",),
          ("d_round_old",), note="旧值是目标五百万/估值三千万，新值是三百五十万/两千八百万"),
    Query("d_q03", "D", "gap", "我团队现在多少人", ("d_team",)),
    Query("d_q04", "D", "self", "你帮我算过哪个产品的定价", ("d_self_pricing",)),
    Query("d_q05", "D", "agreement", "每周一你要给我什么", ("d_agreement_weekly",)),
    Query("d_q06", "D", "gap", "我睡眠怎么样", ("d_insomnia",)),
    Query("d_q07", "D", "temporal", "上个月公司出什么事了", ("d_event_listing", "d_event_lawsuit"),
          note="两条都发生在近期，应都召回"),
    Query("d_q08", "D", "multihop", "我想招人，预算多少，招什么岗", ("d_hire",)),
    Query("d_q09", "D", "gap", "我应酬喝酒行不行", ("d_drink",)),
    Query("d_q10", "D", "abstain", "我孩子在哪个学校", note="用户未婚未育"),
    Query("d_q11", "D", "distractor", "我生日几号", ("d_birthday",),
          note="干扰：多个 persona 都有生日"),
    Query("d_q12", "D", "gap", "我希望你怎么组织内容", ("d_style",)),
    Query("d_q13", "D", "multihop", "我明年想进哪个市场，为什么选它", ("d_expand",)),
    Query("d_q14", "D", "self", "你之前给我写的商业计划书是什么结构", ("d_self_bp",)),
    Query("d_q15", "D", "gap", "我供应商合作多久了，账期多长", ("d_supplier",)),

    # ---------------- Persona E ----------------
    Query("e_q01", "E", "easy", "我在哪个学校读书，大几了", ("e_name",)),
    Query("e_q02", "E", "gap", "我考研准备得怎么样", ("e_exam_plan", "e_exam_school", "e_study_time")),
    Query("e_q03", "E", "distractor", "我女朋友在哪", ("e_gf",),
          note="干扰：e_trip_xian 提到西安"),
    Query("e_q04", "E", "gap", "我家狗上次怎么了", ("e_dog_sick",)),
    Query("e_q05", "E", "temporal", "我国庆去哪玩了，花了多少", ("e_trip_xian", "e_trip_budget")),
    Query("e_q06", "E", "self", "你帮我排查过什么 bug", ("e_self_bug",)),
    Query("e_q07", "E", "agreement", "周日晚上你要帮我做什么", ("e_agreement",)),
    Query("e_q08", "E", "gap", "我兼职一小时多少钱", ("e_parttime",)),
    Query("e_q09", "E", "multihop", "我做课程设计卡在哪了", ("e_project", "e_project_block")),
    Query("e_q10", "E", "abstain", "我实习拿到 offer 了吗", note="对话里从未提过实习"),
    Query("e_q11", "E", "gap", "我打游戏什么水平", ("e_game",)),
    Query("e_q12", "E", "distractor", "我室友都叫什么", ("e_roommate",),
          note="干扰：e_roommate_conflict 只提一个人"),
    Query("e_q13", "E", "gap", "我英语过没过", ("e_english",)),
    Query("e_q14", "E", "gap", "我作息是什么样的", ("e_night",)),
    Query("e_q15", "E", "multihop", "我绩点多少，能保研吗", ("e_gpa",)),
]


def memories_for(persona: str) -> list[Mem]:
    return [m for m in ALL_MEMORIES if m.id.split("_")[0] == persona.lower()]


def queries_for(persona: str) -> list[Query]:
    return [q for q in QUERIES if q.persona == persona]


def self_check() -> list[str]:
    """数据自检：id 唯一、gold 存在、stale 存在、stale 确实被取代。

    测试集本身的错误会让评测结论完全失真，所以每次跑评测前必须先过这一关。
    """
    errs: list[str] = []
    ids = [m.id for m in ALL_MEMORIES]
    if len(ids) != len(set(ids)):
        dup = {i for i in ids if ids.count(i) > 1}
        errs.append(f"记忆 id 重复: {sorted(dup)}")
    idset = set(ids)
    qids = [q.qid for q in QUERIES]
    if len(qids) != len(set(qids)):
        errs.append("query id 重复")
    for q in QUERIES:
        for g in q.gold:
            if g not in idset:
                errs.append(f"{q.qid} 的 gold {g} 不存在")
        for s in q.stale:
            if s not in idset:
                errs.append(f"{q.qid} 的 stale {s} 不存在")
            elif _MEM_BY_ID[s].superseded_by is None:
                errs.append(f"{q.qid} 的 stale {s} 未被任何记忆取代")
        if q.bucket == "abstain" and q.gold:
            errs.append(f"{q.qid} 是 abstain 却标了 gold")
        if q.bucket != "abstain" and not q.gold:
            errs.append(f"{q.qid} 非 abstain 却没有 gold")
    return errs

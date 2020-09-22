use grok;
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};

#[derive(Eq, PartialEq, Hash, Debug)]
enum Token {
    WildCard,
    Val(String),
}

struct TokenVisitor;
impl<'de> Visitor<'de> for TokenVisitor {
    type Value = Token;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if value == "<*>" {
            Ok(Token::WildCard)
        } else {
            Ok(Token::Val(String::from(value)))
        }
    }
}

impl Serialize for Token {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.to_string().as_str())
    }
}

impl<'de> Deserialize<'de> for Token {
    fn deserialize<D>(deserializer: D) -> Result<Self, <D as Deserializer<'de>>::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(TokenVisitor)
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Val(s) => write!(f, "{}", s.as_str()),
            Token::WildCard => write!(f, "{}", "<*>"),
        }
    }
}

impl std::clone::Clone for Token {
    fn clone(&self) -> Self {
        match self {
            Token::WildCard => Token::WildCard,
            Token::Val(s) => Token::Val(s.clone()),
        }
    }
}

#[derive(PartialEq)]
struct GroupSimilarity {
    approximate_similarity: f32,
    exact_similarity: f32,
}

impl core::cmp::PartialOrd for GroupSimilarity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.exact_similarity.partial_cmp(&other.exact_similarity) {
            Some(order) => match order {
                Ordering::Equal => self
                    .approximate_similarity
                    .partial_cmp(&other.approximate_similarity),
                Ordering::Less => Some(Ordering::Less),
                Ordering::Greater => Some(Ordering::Greater),
            },
            None => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogCluster {
    log_tokens: Vec<Token>,
    num_matched: u64,
}

impl fmt::Display for LogCluster {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}, count [{}] ",
            self.log_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<String>>()
                .join(" "),
            self.num_matched
        )
    }
}

impl LogCluster {
    fn new(log_tokens: Vec<Token>) -> LogCluster {
        LogCluster {
            log_tokens,
            num_matched: 1,
        }
    }

    pub fn as_string(&self) -> String {
        self.log_tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(" ")
    }

    fn similarity(&self, log: &[Token]) -> GroupSimilarity {
        let len = self.log_tokens.len() as f32;
        let mut approximate_similarity: f32 = 0.0;
        let mut exact_similarity: f32 = 0.0;

        for (pattern, token) in self.log_tokens.iter().zip(log.iter()) {
            if token == pattern {
                approximate_similarity += 1.0;
                exact_similarity += 1.0;
            } else if *pattern == Token::WildCard {
                approximate_similarity += 1.0;
            }
        }
        GroupSimilarity {
            approximate_similarity: approximate_similarity / len,
            exact_similarity: exact_similarity / len,
        }
    }

    fn add_log(&mut self, log: &[Token]) {
        for i in 0..log.len() {
            let token = &self.log_tokens[i];
            if token != &Token::WildCard {
                let other_token = &log[i];
                if token != other_token {
                    self.log_tokens[i] = Token::WildCard;
                }
            }
        }
        self.num_matched += 1;
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Leaf {
    log_groups: Vec<LogCluster>,
}

struct GroupAndSimilarity {
    group_index: usize,
    similarity: GroupSimilarity,
}

impl Leaf {
    fn best_group(&self, log_tokens: &[Token]) -> Option<GroupAndSimilarity> {
        let mut max_similarity = match self.log_groups.get(0) {
            Some(group) => group.similarity(log_tokens),
            None => return None,
        };
        let mut group_index: usize = 0;
        for i in 1..self.log_groups.len() {
            let group = self.log_groups.get(i).unwrap();
            let similarity = group.similarity(log_tokens);
            if similarity > max_similarity {
                max_similarity = similarity;
                group_index = i;
            }
        }
        Some(GroupAndSimilarity {
            group_index,
            similarity: max_similarity,
        })
    }

    fn add_to_group(
        &mut self,
        group: Option<GroupAndSimilarity>,
        min_similarity: &f32,
        log_tokens: &[Token],
    ) {
        match group {
            Some(gas) => {
                if gas.similarity.approximate_similarity < *min_similarity {
                    self.log_groups.push(LogCluster::new(log_tokens.to_vec()))
                } else {
                    self.log_groups
                        .get_mut(gas.group_index)
                        .expect(format!("bad log group index [{}]", gas.group_index).as_str())
                        .add_log(log_tokens)
                }
            }
            None => self.log_groups.push(LogCluster::new(log_tokens.to_vec())),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Inner {
    children: HashMap<Token, Node>,
    depth: usize,
}

#[derive(Debug, Serialize, Deserialize)]
enum Node {
    Inner(Inner),
    Leaf(Leaf),
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut str = String::new();
        match self {
            Node::Inner(node) => {
                for (k, v) in node.children.iter() {
                    str += &format!(
                        "{}Token: {} -> Children [{}]\n",
                        " ".repeat(node.depth),
                        k,
                        v
                    )
                    .to_string();
                }
            }
            Node::Leaf(node) => {
                for lg in node.log_groups.iter() {
                    str += &format!("group [{}]", lg).to_string();
                }
            }
        }
        write!(f, "[\n{}\n]", str)
    }
}

impl Node {
    fn log_groups(&self) -> Vec<&LogCluster> {
        match self {
            Node::Leaf(leaf) => leaf
                .log_groups
                .iter()
                .map(|n| n.borrow())
                .collect::<Vec<&LogCluster>>(),
            Node::Inner(inner) => inner
                .children
                .values()
                .flat_map(|n| n.log_groups())
                .collect::<Vec<&LogCluster>>(),
        }
    }

    fn inner(depth: usize) -> Node {
        Node::Inner(Inner {
            children: HashMap::new(),
            depth,
        })
    }

    fn leaf() -> Node {
        Node::Leaf(Leaf { log_groups: vec![] })
    }

    fn add_child_recur(
        &mut self,
        depth: usize,
        max_depth: &u16,
        max_children: &u16,
        min_similarity: &f32,
        log_tokens: &[Token],
    ) {
        let token = match &log_tokens[depth] {
            Token::Val(s) => {
                if s.chars().any(|c| c.is_numeric()) {
                    Token::WildCard
                } else {
                    Token::Val(s.clone())
                }
            }
            Token::WildCard => Token::WildCard,
        };
        if depth == log_tokens.len() - 1 || depth == *max_depth as usize {
            if let Node::Inner(node) = self {
                let child = node.children.entry(token).or_insert(Node::leaf());
                if let Node::Leaf(leaf) = child {
                    let best_group = leaf.best_group(log_tokens);
                    leaf.add_to_group(best_group, min_similarity, log_tokens);
                }
            }
            return;
        }
        match self {
            Node::Inner(inner) => {
                let child = if !inner.children.contains_key(&token)
                    && inner.children.len() > *max_children as usize
                {
                    inner
                        .children
                        .entry(Token::WildCard)
                        .or_insert(Node::inner(depth + 1))
                } else {
                    inner
                        .children
                        .entry(token)
                        .or_insert(Node::inner(depth + 1))
                };
                child.add_child_recur(
                    depth + 1,
                    max_depth,
                    max_children,
                    min_similarity,
                    log_tokens,
                );
            }
            Node::Leaf(leaf) => {
                let best_group = leaf.best_group(log_tokens);
                leaf.add_to_group(best_group, min_similarity, log_tokens);
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DrainTree {
    root: HashMap<usize, Node>,
    max_depth: u16,
    max_children: u16,
    min_similarity: f32,
    overall_pattern_str: Option<String>,
    #[serde(skip)]
    overall_pattern: Option<grok::Pattern>,
    drain_field: Option<String>,
    #[serde(skip)]
    filter_patterns: Vec<grok::Pattern>,
    filter_patterns_str: Vec<String>,
}

impl Display for DrainTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut str = String::new();
        for (k, v) in self.root.iter() {
            str += &format!("Len: {} -> [ {} ]\n", k, v).to_string();
        }
        write!(f, "[\n{}\n]", str)
    }
}

impl DrainTree {
    pub fn new() -> Self {
        DrainTree {
            root: HashMap::new(),
            filter_patterns_str: vec![],
            filter_patterns: vec![],
            max_depth: 5,
            max_children: 100,
            min_similarity: 0.5,
            overall_pattern: None,
            overall_pattern_str: None,
            drain_field: None,
        }
    }

    pub fn max_depth(mut self, max_depth: u16) -> Self {
        self.max_depth = max_depth;
        self
    }

    pub fn max_children(mut self, max_children: u16) -> Self {
        self.max_children = max_children;
        self
    }

    pub fn min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity;
        self
    }

    pub fn filter_patterns(mut self, filter_patterns: Vec<&str>) -> Self {
        self.filter_patterns_str = filter_patterns
            .iter()
            .map(|s| String::from(*s))
            .collect::<Vec<String>>();
        self
    }

    pub fn log_pattern(mut self, overall_pattern: &str, drain_field: &str) -> Self {
        self.overall_pattern_str = Some(String::from(overall_pattern));
        self.drain_field = Some(String::from(drain_field));
        self
    }

    // ugh, I don't like this. Would it be better to have a separate struct for building
    // and then returning a DrainTree?
    pub fn build_patterns(mut self, grok: &mut grok::Grok) -> Self {
        if let Some(pattern_str) = &self.overall_pattern_str {
            self.overall_pattern = Some(
                grok.compile(pattern_str.as_str(), true)
                    .expect("poorly formatted overall_pattern"),
            );
        }
        let mut filter_patterns = Vec::with_capacity(*&self.filter_patterns_str.len());
        for pattern in &self.filter_patterns_str {
            if let Ok(c) = grok.compile(pattern.as_str(), true) {
                filter_patterns.push(c);
            }
        }
        self.filter_patterns = filter_patterns;
        self
    }

    fn process(filter_patterns: &Vec<grok::Pattern>, log_line: String) -> Vec<Token> {
        log_line
            .split(' ')
            .map(|t| t.trim())
            .map(|t| {
                match filter_patterns
                    .iter()
                    .map(|p| p.match_against(t))
                    .filter(|o| o.is_some())
                    .next()
                {
                    Some(m) => match m {
                        Some(matches) => match matches.iter().next() {
                            Some((name, _pattern)) => Token::Val(format!("<{}>", name)),
                            None => Token::WildCard,
                        },
                        None => Token::Val(String::from(t)),
                    },
                    None => Token::Val(String::from(t)),
                }
            })
            .collect()
    }

    fn dig_inner_prefix_tree<'a>(
        &self,
        child: &'a Node,
        processed_log: &[Token],
    ) -> Option<&'a LogCluster> {
        let mut current_node = Some(child);
        for t in processed_log {
            if let Some(node) = current_node {
                if let Node::Inner(inner) = node {
                    current_node = inner.children.get(t);
                }
            }
        }
        if let Some(node) = current_node {
            if let Node::Leaf(leaf) = node {
                return match leaf.best_group(processed_log) {
                    Some(gas) => Some(&leaf.log_groups[gas.group_index]),
                    None => None,
                };
            }
        }
        None
    }

    fn log_group_for_tokens(&self, processed_log: &[Token]) -> Option<&LogCluster> {
        match self.root.get(&processed_log.len()) {
            Some(node) => self.dig_inner_prefix_tree(node, processed_log),
            None => Option::None,
        }
    }

    fn apply_overall_pattern(&self, log_line: &str) -> Option<String> {
        match &self.overall_pattern {
            Some(p) => {
                match p.match_against(log_line) {
                    Some(matches) => {
                        match matches.get(self.drain_field.as_ref().expect("illegal state. [overall_pattern] set without [drain_field] set").as_str()) {
                            Some(s) => Option::Some(String::from(s)),
                            None => Option::None
                        }
                    }
                    None => Option::None,
                }
            }
            None => Option::None,
        }
    }

    fn is_compiled(&self) -> bool {
        self.filter_patterns.len() == self.filter_patterns_str.len()
            && (self.overall_pattern.is_some() == self.overall_pattern_str.is_some())
    }

    pub fn log_group(&self, log_line: &str) -> Option<&LogCluster> {
        let processed_line = self.apply_overall_pattern(log_line);
        let tokens = DrainTree::process(
            &self.filter_patterns,
            processed_line.unwrap_or(log_line.to_string()),
        );
        self.log_group_for_tokens(tokens.as_slice())
    }

    pub fn add_log_line(&mut self, log_line: &str) {
        let processed_line = self.apply_overall_pattern(log_line);
        let tokens = DrainTree::process(
            &self.filter_patterns,
            processed_line.unwrap_or(log_line.to_string()),
        );
        let len = tokens.len();
        self.root
            .entry(len)
            .or_insert(Node::inner(0))
            .add_child_recur(
                0,
                &self.max_depth,
                &self.max_children,
                &self.min_similarity,
                tokens.as_slice(),
            );
    }

    pub fn log_groups(&self) -> Vec<&LogCluster> {
        self.root
            .values()
            .flat_map(|n| n.log_groups())
            .collect::<Vec<&LogCluster>>()
    }
}

#[cfg(test)]
mod tests {
    extern crate serde_json;

    const WILDCARD: &str = "<*>";
    use super::*;

    fn tokens_from(strs: &[&str]) -> Vec<Token> {
        let mut v = Vec::with_capacity(strs.len());
        for s in strs.iter() {
            if *s == WILDCARD {
                v.push(Token::WildCard)
            } else {
                v.push(Token::Val(String::from(*s)))
            }
        }
        v
    }

    #[test]
    fn patterns_built() {
        let drain = DrainTree::new();
        assert!(drain.is_compiled());

        let mut g = grok::Grok::with_patterns();

        let filter_patterns = vec!["%{IPV4:ip_address}", "%{NUMBER:user_id}"];
        let drain = DrainTree::new().filter_patterns(filter_patterns);
        assert!(!drain.is_compiled());

        let drain = drain.build_patterns(&mut g);
        assert!(drain.is_compiled());

        let drain = drain.log_pattern(
            "%{NUMBER:id} \\[%{LOGLEVEL:level}\\] %{GREEDYDATA:content}",
            "content",
        );

        assert!(!drain.is_compiled());

        let drain = drain.build_patterns(&mut g);
        assert!(drain.is_compiled());

        let drain = DrainTree::new().log_pattern(
            "%{NUMBER:id} \\[%{LOGLEVEL:level}\\] %{GREEDYDATA:content}",
            "content",
        );
        assert!(!drain.is_compiled());
        let drain = drain.build_patterns(&mut g);
        assert!(drain.is_compiled());
    }

    #[test]
    fn similarity_check() {
        let tokens = tokens_from(&["foo", WILDCARD, "foo", "bar", "baz"]);
        let template = tokens_from(&["foo", "bar", WILDCARD, "bar", "baz"]);
        let group = LogCluster::new(template);
        let similarity = group.similarity(tokens.as_slice());

        assert_eq!(similarity.exact_similarity, 0.6);
        assert_eq!(similarity.approximate_similarity, 0.8);
    }

    #[test]
    fn best_group() {
        let tokens = tokens_from(&["foo", WILDCARD, "foo", "bar", "baz"]);

        let leaf = Leaf {
            log_groups: vec![
                LogCluster::new(tokens_from(&["foo", "bar", WILDCARD, "bar", "baz"])),
                LogCluster::new(tokens_from(&["foo", "bar", "other", "bar", "baz"])),
                LogCluster::new(tokens_from(&["a", "b", WILDCARD, "c", "baz"])),
            ],
        };

        let best_group = leaf
            .best_group(tokens.as_slice())
            .expect("missing best group");

        assert_eq!(best_group.group_index, 0);
        assert_eq!(best_group.similarity.exact_similarity, 0.6);
        assert_eq!(best_group.similarity.approximate_similarity, 0.8);

        let leaf = Leaf {
            log_groups: vec![
                LogCluster::new(tokens_from(&["a", "b", WILDCARD, "c", "baz"])),
                LogCluster::new(tokens_from(&["foo", "bar", "other", "bar", "baz"])),
            ],
        };
        let best_group = leaf
            .best_group(tokens.as_slice())
            .expect("missing best group");

        assert_eq!(best_group.group_index, 1);
        assert_eq!(best_group.similarity.exact_similarity, 0.6);
        assert_eq!(best_group.similarity.approximate_similarity, 0.6);
    }

    #[test]
    fn add_group() {
        let tokens = tokens_from(&["foo", WILDCARD, "foo", "bar", "baz"]);
        let min_sim = 0.5;
        let leaf_ctor = || Leaf {
            log_groups: vec![
                LogCluster::new(tokens_from(&["foo", "bar", WILDCARD, "bar", "baz"])),
                LogCluster::new(tokens_from(&["foo", "bar", "other", "bar", "baz"])),
                LogCluster::new(tokens_from(&["a", "b", WILDCARD, "c", "baz"])),
            ],
        };

        // Add new group as no similarity was provided
        {
            let mut leaf = leaf_ctor();
            leaf.add_to_group(Option::None, &min_sim, tokens.as_slice());
            assert_eq!(leaf.log_groups.len(), 4);
        }
        // lower than minimum similarity, new group is added
        {
            let mut leaf = leaf_ctor();
            leaf.add_to_group(
                Option::Some(GroupAndSimilarity {
                    group_index: 1,
                    similarity: GroupSimilarity {
                        exact_similarity: 0.1,
                        approximate_similarity: 0.1,
                    },
                }),
                &min_sim,
                tokens.as_slice(),
            );
            assert_eq!(leaf.log_groups.len(), 4);
        }

        {
            let mut leaf = leaf_ctor();
            leaf.add_to_group(Option::None, &min_sim, tokens.as_slice());
            assert_eq!(leaf.log_groups.len(), 4);
        }
        // adds new group and adjusts stored tokens
        {
            let mut leaf = leaf_ctor();
            leaf.add_to_group(
                Option::Some(GroupAndSimilarity {
                    group_index: 0,
                    similarity: GroupSimilarity {
                        exact_similarity: 0.6,
                        approximate_similarity: 0.6,
                    },
                }),
                &min_sim,
                tokens.as_slice(),
            );
            assert_eq!(leaf.log_groups.len(), 3);
            assert_eq!(
                leaf.log_groups[0].log_tokens,
                tokens_from(&["foo", WILDCARD, WILDCARD, "bar", "baz"])
            );
        }
    }

    #[test]
    fn log_clustering() {
        let logs = vec![
            "1 [INFO] user 3 called 192.0.0.1",
            "2 [INFO] user 2 called 127.0.0.1",
            "3 [DEBUG] something uninteresting happened",
            "4 [INFO] user 4 called 10.0.0.1",
        ];

        let mut g = grok::Grok::with_patterns();

        let filter_patterns = vec!["%{IPV4:ip_address}", "%{NUMBER:user_id}"];

        let mut drain = DrainTree::new()
            .filter_patterns(filter_patterns)
            .max_depth(4)
            .max_children(100)
            .min_similarity(0.5)
            .log_pattern(
                "%{NUMBER:id} \\[%{LOGLEVEL:level}\\] %{GREEDYDATA:content}",
                "content",
            )
            .build_patterns(&mut g);

        for log in logs {
            drain.add_log_line(log);
        }
        assert_eq!(
            drain
                .log_group(&"10 [INFO] user 40 called 192.168.10.2")
                .expect("missing expected log group")
                .as_string(),
            "user <user_id> called <ip_address>"
        );
        assert_eq!(
            drain
                .log_group(&"2 [INFO] something uninteresting happened")
                .expect("missing expected log group")
                .as_string(),
            "something uninteresting happened"
        );
    }

    #[test]
    fn dump_and_load() {
        let logs = vec![
            "1 [INFO] user 3 called 192.0.0.1",
            "2 [INFO] user 2 called 127.0.0.1",
            "3 [DEBUG] something uninteresting happened",
            "4 [INFO] user 4 called 10.0.0.1",
        ];

        let mut g = grok::Grok::with_patterns();

        let filter_patterns = vec!["%{IPV4:ip_address}", "%{NUMBER:user_id}"];

        let mut drain = DrainTree::new()
            .filter_patterns(filter_patterns)
            .max_depth(4)
            .max_children(100)
            .min_similarity(0.5)
            .log_pattern(
                "%{NUMBER:id} \\[%{LOGLEVEL:level}\\] %{GREEDYDATA:content}",
                "content",
            )
            .build_patterns(&mut g);

        for log in logs {
            drain.add_log_line(log);
        }

        let serialized = serde_json::to_string(&drain).expect("serialization failure");

        let other: DrainTree = serde_json::from_str(serialized.as_str()).unwrap();
        let other = other.build_patterns(&mut g);

        assert_eq!(
            other
                .log_group(&"10 [INFO] user 40 called 192.168.10.2")
                .expect("missing expected log group")
                .as_string(),
            "user <user_id> called <ip_address>"
        );
        assert_eq!(
            other
                .log_group(&"2 [INFO] something uninteresting happened")
                .expect("missing expected log group")
                .as_string(),
            "something uninteresting happened"
        );
    }
}

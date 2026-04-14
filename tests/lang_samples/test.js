const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        const data = await response.json();
        return data;
    } catch (err) {
        console.log("Error:", err);
        return null;
    }
};

function processItems(items) {
    var result = [];
    for (let i = 0; i < items.length; i++) {
        if (items[i].active) {
            const { name, score } = items[i];
            result.push({ name, score: score * 1.1 });
        }
    }
    return result.sort((a, b) => b.score - a.score);
}

const dangerous = () => eval("alert('xss')");

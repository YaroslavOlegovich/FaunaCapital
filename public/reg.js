// Получаем ссылки на кнопки
const registerButton = document.getElementById('register-button');
const loginButton = document.getElementById('login-button');
const homeButton = document.getElementById('home-button');

// Обработчики событий для кнопок
registerButton.addEventListener('click', () => {
    // Обработка кнопки "Зарегистрироваться"
    window.location.href = 'register.html'; // или другая логика
});

loginButton.addEventListener('click', () => {
    // Обработка кнопки "Войти"
    window.location.href = 'login.html'; // или другая логика
});

homeButton.addEventListener('click', () => {
    // Обработка кнопки "На главную"
    window.location.href = 'index.html'; // или другая логика
});